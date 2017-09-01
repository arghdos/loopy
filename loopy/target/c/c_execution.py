from __future__ import division, with_statement, absolute_import

__copyright__ = "Copyright (C) 2017 Nick Curtis"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tempfile
import cgen
import os

from loopy.target.execution import (KernelExecutorBase, _KernelInfo,
                                    ExecutionWrapperGeneratorBase)
from loopy.target.c import CTarget
from pytools import memoize_method
from pytools.py_codegen import (Indentation)
from codepy.toolchain import guess_toolchain
from codepy.jit import compile_from_string
import six
import weakref
import ctypes

import numpy as np

import logging
logger = logging.getLogger(__name__)


class CExecutionWrapperGenerator(ExecutionWrapperGeneratorBase):

    """
    Specialized form of the :class:`ExecutionWrapperGeneratorBase` for
    pyopencl execution
    """

    def __init__(self):
        system_args = ["_lpy_c_kernels"]
        super(CExecutionWrapperGenerator, self).__init__(system_args)

    def python_dtype_str(self, dtype):
        if np.dtype(str(dtype)).isbuiltin:
            return "_lpy_np."+dtype.name
        raise Exception('dtype: {} not recognized'.format(dtype))

    # {{{ handle non numpy arguements

    def handle_non_numpy_arg(self, gen, arg):
        pass

    # }}}

    # {{{ handle allocation of unspecified arguements

    def handle_alloc(self, gen, arg, kernel_arg, strify, skip_arg_checks):
        """
        Handle allocation of non-specified arguements for C-execution
        """
        from pymbolic import var

        num_axes = len(arg.unvec_shape)
        for i in range(num_axes):
            gen("_lpy_shape_%d = %s" % (i, strify(arg.unvec_shape[i])))

        itemsize = kernel_arg.dtype.numpy_dtype.itemsize
        for i in range(num_axes):
            gen("_lpy_strides_%d = %s" % (i, strify(
                itemsize*arg.unvec_strides[i])))

        if not skip_arg_checks:
            for i in range(num_axes):
                gen("assert _lpy_strides_%d > 0, "
                    "\"'%s' has negative stride in axis %d\""
                    % (i, arg.name, i))

        sym_strides = tuple(
            var("_lpy_strides_%d" % i)
            for i in range(num_axes))

        sym_shape = tuple(
            var("_lpy_shape_%d" % i)
            for i in range(num_axes))

        # find order of array
        order = "'C'" if arg.unvec_strides[-1] == 1 else "'F'"

        gen("%(name)s = _lpy_np.empty(%(shape)s, "
            "%(dtype)s, order=%(order)s)"
            % dict(
                name=arg.name,
                shape=strify(sym_shape),
                dtype=self.python_dtype_str(
                    kernel_arg.dtype.numpy_dtype),
                order=order))

        # check strides
        if not skip_arg_checks:
            gen("assert %(strides)s == %(name)s.strides, "
                "'Strides of loopy created array %(name)s, "
                "do not match expected.'" %
                dict(name=arg.name,
                     strides=strify(sym_strides)))
            for i in range(num_axes):
                gen("del _lpy_shape_%d" % i)
                gen("del _lpy_strides_%d" % i)
            gen("")

    # }}}

    def target_specific_preamble(self, gen):
        """
        Add default C-imports to preamble
        """
        gen.add_to_preamble("import numpy as _lpy_np")

    def initialize_system_args(self, gen):
        """
        Initializes possibly empty system arguements
        """
        pass

    # {{{ generate invocation

    def generate_invocation(self, gen, kernel_name, args):
        gen("for knl in _lpy_c_kernels:")
        with Indentation(gen):
            gen('knl({args})'.format(
                args=", ".join(args)))
    # }}}

    # {{{

    def generate_output_handler(
            self, gen, options, kernel, implemented_data_info):

        from loopy.kernel.data import KernelArgument

        if options.return_dict:
            gen("return None, {%s}"
                % ", ".join("\"%s\": %s" % (arg.name, arg.name)
                            for arg in implemented_data_info
                            if issubclass(arg.arg_class, KernelArgument)
                            if arg.base_name in kernel.get_written_variables()))
        else:
            out_args = [arg
                        for arg in implemented_data_info
                        if issubclass(arg.arg_class, KernelArgument)
                        if arg.base_name in kernel.get_written_variables()]
            if out_args:
                gen("return None, (%s,)"
                    % ", ".join(arg.name for arg in out_args))
            else:
                gen("return None, ()")

    # }}}

    def generate_host_code(self, gen, codegen_result):
        pass

    def get_arg_pass(self, arg):
        return arg.name


class CCompiler(object):

    """
    The compiler module handles invocation of compilers to generate a shared lib
    using codepy, which can subsequently be loaded via ctypes.

    The general strategy here is as follows:


    1.  A :class:`codepy.Toolchain` is guessed from distutils.
        The user may override any flags obtained therein by passing in arguements
        to cc, cflags, etc.

    2.  The kernel source is built into and object first, then made into a shared
        library using :meth:`codepy.jit.compile_from_string`, which additionally
        handles caching

    3.  The resulting shared library is turned into a :class:`ctypes.CDLL`
        to enable calling by the invoker generated by, e.g.,
        :class:`CExecutionWrapperGenerator`
    """

    def __init__(self, toolchain=None,
                 cc='gcc', cflags='-std=c99 -O3 -fPIC'.split(),
                 ldflags='-shared'.split(), libraries=[],
                 include_dirs=[], library_dirs=[], defines=[],
                 source_suffix='c'):

        # try to get a default toolchain
        # or subclass supplied version if available
        self.toolchain = guess_toolchain() if toolchain is None else toolchain
        self.source_suffix = source_suffix
        if toolchain is None:
            # copy in all differing values
            diff = {'cc': cc,
                    'cflags': cflags,
                    'ldflags': ldflags,
                    'libraries': libraries,
                    'include_dirs': include_dirs,
                    'library_dirs': library_dirs,
                    'defines': defines}
            # filter empty and those equal to toolchain defaults
            diff = dict((k, v) for k, v in six.iteritems(diff)
                    if v and
                    not hasattr(self.toolchain, k) or
                    getattr(self.toolchain, k) != v)
            self.toolchain = self.toolchain.copy(**diff)
        self.tempdir = tempfile.mkdtemp(prefix="tmp_loopy")
        self.source_suffix = source_suffix

    def _tempname(self, name):
        """Build temporary filename path in tempdir."""
        return os.path.join(self.tempdir, name)

    @memoize_method
    def _build_obj(self, name, code, source_name,
                   debug=False, wait_on_error=None, debug_recompile=True):
        """Compile code, and build object file"""
        logger.debug(code)

        # build object
        obj_checksum, _, obj_file, recompiled = \
            compile_from_string(self.toolchain, name, code, source_name,
                                self.tempdir, debug, wait_on_error,
                                debug_recompile, True)
        if not recompiled:
            logger.debug('Kernel {} compiled from source'.format(name))

        return obj_checksum, obj_file

    @memoize_method
    def _build_lib(self, name, obj_file, debug=False, wait_on_error=None,
                   debug_recompile=True):
        """Build and load shared library from object file"""

        if not isinstance(obj_file, tuple):
            obj_file = (obj_file,)

        # read obj files in to get "source"
        obj = []
        obj_name = []
        from os.path import basename
        for o in obj_file:
            with open(o, 'rb') as file:
                obj.append(file.read())
            obj_name.append(basename(o))

        # build object
        so_checksum, _, so_file, recompiled = \
            compile_from_string(self.toolchain, name, obj, obj_name,
                                self.tempdir, debug, wait_on_error,
                                debug_recompile, object=False,
                                source_is_binary=True)
        if not recompiled:
            logger.debug('Kernel {} compiled from source'.format(name))

        return so_checksum, ctypes.CDLL(so_file)

    def build(self, name, code, debug=False, wait_on_error=None,
              debug_recompile=True):
        """Compile code, build and load shared library."""

        # build object
        _, obj_file = self._build_obj(
            name, code, self._tempname('code.' + self.source_suffix),
            debug=debug, wait_on_error=wait_on_error,
            debug_recompile=debug_recompile)

        # and create library
        _, lib = self._build_lib(name, obj_file, debug=debug,
                              wait_on_error=wait_on_error,
                              debug_recompile=debug_recompile)

        # and return compiled
        return lib


class CPlusPlusCompiler(CCompiler):
    """Subclass of CCompiler to invoke a C++ compiler."""

    def __init__(self, cc='g++', cflags='',
                 ldflags=[], libraries=[],
                 include_dirs=[], library_dirs=[], defines=[],
                 source_suffix='cpp'):

        super(CPlusPlusCompiler, self).__init__(
            cc=cc, cflags=cflags, ldflags=ldflags, libraries=libraries,
            include_dirs=include_dirs, library_dirs=library_dirs,
            defines=defines, source_suffix=source_suffix)


class CompiledCKernel(object):

    """
    A CompiledCKernel wraps a loopy kernel, compiling it and loading the
    result as a shared library, and provides access to the kernel as a
    ctypes function object, wrapped by the __call__ method, which attempts
    to automatically map argument types.
    """

    def __init__(self, knl, dev_code='', host_code='',
                 host_name='', target=CTarget(), comp=None):
        assert isinstance(target, CTarget)
        self.target = target
        self.knl = knl
        # get code and build
        self.dev_code = dev_code
        self.host_code = host_code
        self.host_name = host_name
        self.code = self._get_code()
        self.comp = comp or CCompiler()
        self.dll = self.comp.build(self.knl.name, self.code)

        # get the function declaration for interface with ctypes
        self.func_decl = self._get_extractor()
        self.func_decl(knl.ast)
        self.func_decl = self.func_decl.decls[0]
        self._arg_info = []
        # TODO knl.args[:].dtype is sufficient
        self._visit_func_decl(self.func_decl)
        self.name = self.knl.name
        restype = self.func_decl.subdecl.typename
        if restype == 'void':
            self.restype = None
        else:
            raise ValueError('Unhandled restype %r' % (restype, ))
        self._fn = getattr(self.dll, self._get_linking_name())
        self._fn.restype = self.restype
        self._fn.argtypes = [ctype for name, ctype in self._arg_info]
        self._prepared_call_cache = weakref.WeakKeyDictionary()

    def _get_linking_name(self):
        """ return device program name for C-kernel """
        return self.knl.name

    def _get_code(self):
        """ No 'host' for C-only """
        return self.dev_code

    def _get_extractor(self):
        """ Returns the correct function decl extractor depending on target
            type"""
        from loopy.target.c import CFunctionDeclExtractor
        return CFunctionDeclExtractor()

    def __call__(self, *args):
        """Execute kernel with given args mapped to ctypes equivalents."""
        args_ = []
        for arg, arg_t in zip(args, self._fn.argtypes):
            if hasattr(arg, 'ctypes'):
                if arg.size == 0:
                    # TODO eliminate unused arguments from kernel
                    arg_ = arg_t(0.0)
                else:
                    arg_ = arg.ctypes.data_as(arg_t)
            else:
                arg_ = arg_t(arg)
            args_.append(arg_)
        self._fn(*args_)

    def _append_arg(self, name, dtype, pointer=False):
        """Append arg info to current argument list."""
        self._arg_info.append((
            name,
            self._dtype_to_ctype(dtype, pointer=pointer)
        ))

    def _visit_const(self, node):
        """Visit const arg of kernel."""
        if isinstance(node.subdecl, cgen.RestrictPointer):
            self._visit_pointer(node.subdecl)
        else:
            pod = node.subdecl  # type: cgen.POD
            self._append_arg(pod.name, pod.dtype)

    def _visit_pointer(self, node):
        """Visit pointer argument of kernel."""
        pod = node.subdecl  # type: cgen.POD
        self._append_arg(pod.name, pod.dtype, pointer=True)

    def _visit_func_decl(self, func_decl):
        """Visit nodes of function declaration of kernel."""
        for i, arg in enumerate(func_decl.arg_decls):
            if isinstance(arg, cgen.Const):
                self._visit_const(arg)
            elif isinstance(arg, cgen.RestrictPointer):
                self._visit_pointer(arg)
            else:
                raise ValueError('unhandled type for arg %r' % (arg, ))

    def _dtype_to_ctype(self, dtype, pointer=False):
        """Map NumPy dtype to equivalent ctypes type."""
        target = self.target  # type: CTarget
        registry = target.get_dtype_registry().wrapped_registry
        typename = registry.dtype_to_ctype(dtype)
        typename = {'unsigned': 'uint'}.get(typename, typename)
        basetype = getattr(ctypes, 'c_' + typename)
        if pointer:
            return ctypes.POINTER(basetype)
        return basetype


class CKernelExecutor(KernelExecutorBase):

    """An object connecting a kernel to a :class:`CompiledKernel`
    for execution.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, kernel, invoker=CExecutionWrapperGenerator(),
                 compiler=None):
        """
        :arg kernel: may be a loopy.LoopKernel, a generator returning kernels
            (a warning will be issued if more than one is returned). If the
            kernel has not yet been loop-scheduled, that is done, too, with no
            specific arguments.
        """

        self.compiler = compiler if compiler else CCompiler()
        super(CKernelExecutor, self).__init__(kernel, invoker=invoker)

    def get_compiled(self, *args, **kwargs):
        return CompiledCKernel(*args, **kwargs)

    @memoize_method
    def kernel_info(self, arg_to_dtype_set=frozenset(), all_kwargs=None):
        kernel = self.get_typed_and_scheduled_kernel(arg_to_dtype_set)

        from loopy.codegen import generate_code_v2
        codegen_result = generate_code_v2(kernel)

        dev_code = codegen_result.device_code()

        if self.kernel.options.write_cl:
            output = dev_code
            if self.kernel.options.highlight_cl:
                output = self.get_highlighted_code(output)

            if self.kernel.options.write_cl is True:
                print(output)
            else:
                with open(self.kernel.options.write_cl, "w") as outf:
                    outf.write(output)

        if self.kernel.options.edit_cl:
            from pytools import invoke_editor
            dev_code = invoke_editor(dev_code, "code.c")

        c_kernels = []
        for dp in codegen_result.device_programs:
            c_kernels.append(self.get_compiled(
                             dp, dev_code=dev_code,
                             host_code=codegen_result.host_code(),
                             host_name=codegen_result.host_program.name,
                             target=self.kernel.target,
                             comp=self.compiler))

        return _KernelInfo(
            kernel=kernel,
            c_kernels=c_kernels,
            implemented_data_info=codegen_result.implemented_data_info,
            invoker=self.invoker(kernel, codegen_result))

    # }}}

    def __call__(self, *args, **kwargs):
        """
        :returns: ``(None, output)`` the output is a tuple of output arguments
            (arguments that are written as part of the kernel). The order is given
            by the order of kernel arguments. If this order is unspecified
            (such as when kernel arguments are inferred automatically),
            enable :attr:`loopy.Options.return_dict` to make *output* a
            :class:`dict` instead, with keys of argument names and values
            of the returned arrays.
        """

        kwargs = self.packing_controller.unpack(kwargs)

        kernel_info = self.kernel_info(self.arg_to_dtype_set(kwargs))

        return kernel_info.invoker(
            kernel_info.c_kernels, *args, **kwargs)
