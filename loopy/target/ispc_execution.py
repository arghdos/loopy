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

from loopy.target.c.c_execution import (CKernelExecutor, CCompiler,
                                        CExecutionWrapperGenerator, CompiledCKernel)
from loopy.target.ispc import ISPCCFunctionDeclExtractor
from codepy.toolchain import (GCCLikeToolchain, Toolchain,
                              _guess_toolchain_kwargs_from_python_config,
                              call_capture_output, CompileError)
import cgen
import ctypes
import os


class ISPCToolchain(GCCLikeToolchain):

    def get_version_tuple(self):
        ver = self.get_version()
        words = ver.split()
        numbers = words[4].split(".")

        result = []
        for n in numbers:
            try:
                result.append(int(n))
            except ValueError:
                # not an integer? too bad.
                break

        return tuple(result)

    def get_cc(self, files, obj=True):
        if obj and all(f.endswith('.ispc') for f in files) or not files:
            return self.cc
        elif all(f.endswith('.cpp') for f in files):
            return self.cpp
        else:
            return self.ld

    def get_dependencies(self, source_files):
        building_obj = True
        if all(source.endswith('.o') for source in source_files):
            # object file
            building_obj = False

        cc = self.get_cc(source_files, obj=building_obj)
        cflags = self.cflags if cc == self.cc else self.cppflags

        from codepy.tools import join_continued_lines
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(prefix='loopy') as tempfile:
            depends = ['-MMM', tempfile.name] if cc == 'ispc' else ['-M']
            result, stdout, stderr = call_capture_output(
                [cc]
                + depends
                + ["-D%s" % define for define in self.defines]
                + ["-U%s" % undefine for undefine in self.undefines]
                + ["-I%s" % idir for idir in self.include_dirs]
                + cflags
                + source_files
            )

            if result != 0:
                raise CompileError("getting dependencies failed: " + stderr)

            lines = join_continued_lines(tempfile.read().decode().split("\n"))

        from pytools import flatten
        return set(flatten(
            line.split()[2:] for line in lines))

    def _cmdline(self, file, obj=False):
        flags = self.cflags
        cc = self.get_cc(file, obj=obj)
        if cc != self.cc:
            # fix flags
            flags = self.cppflags + (['-c'] if obj else [])

            # check we don't have mixed extensions
            def __get_ext(f):
                return f[f.rindex('.') + 1:]

            ext = __get_ext(file[0])
            if not all(f.endswith(ext) for f in file):
                ftypes = set([__get_ext(f) for f in file])
                raise CompileError("Can't compile mixed filetypes: {}".format(
                    ', '.join(ftypes)))

        if obj:
            ld_options = []
            link = []
        else:
            ld_options = self.ldflags
            link = ["-L%s" % ldir for ldir in self.library_dirs]
            link.extend(["-l%s" % lib if not lib == '-fopenmp' else '-fopenmp'
                         for lib in self.libraries])
        return (
            [cc]
            + flags
            + ld_options
            + ["-D%s" % define for define in self.defines]
            + ["-U%s" % undefine for undefine in self.undefines]
            + ["-I%s" % idir for idir in self.include_dirs]
            + file
            + link
        )

    def abi_id(self):
        return Toolchain.abi_id(self) + [self._cmdline([])]

    def with_optimization_level(self, level, debug=False, **extra):
        def remove_prefix(l, prefix):
            return [f for f in l if not f.startswith(prefix)]

        cflags = self.cflags
        for pfx in ["-O", "-g", "-DNDEBUG"]:
            cflags = remove_prefix(cflags, pfx)

        if level == "debug":
            oflags = ["-g"]
        else:
            oflags = ["-O%d" % level, "-DNDEBUG"]

        return self.copy(cflags=cflags + oflags)


class ISPCCompiler(CCompiler):
    """Subclass of Compiler to invoke the ispc compiler."""

    def __init__(self, use_openmp=True, target_name='', vector_width=None,
                 addressing_width=None, **kwargs):
        """    An interface for intel's ISPC compiler

        Parameters
        ---------
        use_openmp: bool [True]
            Use openmp to parallelize over cores.  If false, pthreads will be used
        target_name: ['sse2', 'sse4', 'avx1', 'avx2', 'generic']
            The vectorization target .  If not supplied the system default will be
            used
        vector_width: int [None]
            The vector width to use for ispc-vectorization.  If not supplied this
            will use the default vector width as determined by ispc
        addressing_width: int [32, 64]
            Select 32- or 64-bit addressing. (Note that 32-bit addressing
            calculations are done by default, even on 64-bit target architectures.)
        """

        target_flags = [target_name, vector_width, addressing_width]
        if not any(target_flags):
            target_flags = ['--target', 'host']
        elif any(target_flags):
            # first find the default system target to fill in gaps
            if not all(target_flags):
                import re
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(prefix='loopy') as tempfile:
                    tempfile.write(b'void test(){} \n')
                    tempfile.flush()
                    _, _, stderr = call_capture_output((['ispc', tempfile.name]))
            # search output
                for line in stderr.split('\n'):
                    match = re.search(r'\"([\w\d]+)-i(\d+)x(\d+)\"', line)
                    if match:
                        # find defaults, and construct target
                        target, addressing, width = match.groups()
                        if not target_name:
                            target_name = target
                        if not vector_width:
                            vector_width = width
                        if not addressing_width:
                            addressing_width = addressing
                        break
            # and construct the user supplied / default target
            target_flags = ['--target', '{0}-i{1}x{2}'.format(
                target_name, addressing_width, vector_width)]

        toolchain_defaults = _guess_toolchain_kwargs_from_python_config()
        cppflags = ['-O3', '-fPIC', ('-fopenmp' if use_openmp else 'pthread')]
        toolchain_kwargs = dict(
            cc='ispc',
            ldflags=['-shared'],
            libraries=['-fopenmp' if use_openmp else 'pthread', 'stdc++'],
            cflags=['-O3', '--pic'] + target_flags,
            include_dirs=toolchain_defaults["include_dirs"],
            library_dirs=toolchain_defaults["library_dirs"],
            so_ext=toolchain_defaults["so_ext"],
            o_ext=toolchain_defaults["o_ext"],
            defines=toolchain_defaults["defines"] + [
                'ISPC_USE_OMP' if use_openmp else 'ISPC_USE_PTHREADS'],
            undefines=toolchain_defaults["undefines"],
            ld='gcc',
            cpp='g++',
            cppflags=cppflags
        )
        defaults = {'toolchain': ISPCToolchain(**toolchain_kwargs),
                    'source_suffix': 'ispc',
                    }

        # and create
        super(ISPCCompiler, self).__init__(**defaults)

    def build(self, name, code, debug=False, wait_on_error=None,
              debug_recompile=True):
        """Compile code, build and load shared library."""

        # build object
        _, obj_file = self._build_obj(
            name, code, self._tempname('code.' + self.source_suffix),
            debug=debug, wait_on_error=wait_on_error,
            debug_recompile=debug_recompile)

        # find the task sys
        result, stdout, stderr = call_capture_output(
            (['which', 'ispc']))
        if result != 0:
            raise CompileError("Could not find ispc executable: " + stderr)
        tasksys = os.path.realpath(
            os.path.join(os.path.dirname(stdout), 'examples', 'tasksys.cpp'))

        # build tasksys obj
        with open(tasksys, 'r') as file:
            tasksys = file.read()

        _, ts_obj_file = self._build_obj(
            'tasksys', tasksys, self._tempname('tasksys.cpp'),
            debug=debug, wait_on_error=wait_on_error,
            debug_recompile=debug_recompile)

        # now call the regular build method with the tasksys code inserted into
        # or compilation process

        # and create library
        _, lib = self._build_lib(name, (obj_file, ts_obj_file),
                              debug=debug,
                              wait_on_error=wait_on_error,
                              debug_recompile=debug_recompile)

        return lib


class ISPCKernelExecutor(CKernelExecutor):

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

        # find the vector width for the kernel, if applicable
        from loopy.kernel.data import LocalIndexTag
        from six import iteritems
        lsize = None
        for iname, tag in iteritems(kernel.iname_to_tag):
            if isinstance(tag, LocalIndexTag):
                _, (lsize,) = kernel.get_grid_size_upper_bounds_as_exprs()

        if compiler and compiler.vector_width is None:
            compiler = compiler.copy(vector_width=lsize)
        self.compiler = compiler if compiler else ISPCCompiler(vector_width=lsize)

        super(ISPCKernelExecutor, self).__init__(
            kernel, invoker=invoker, compiler=self.compiler)

    def get_compiled(self, *args, **kwargs):
        return CompiledISPCKernel(*args, **kwargs)


class CompiledISPCKernel(CompiledCKernel):

    def _get_linking_name(self):
        """ return host program name for ISPC-kernel """
        return self.host_name

    def _get_code(self):
        # need to include the launcher
        return '\n'.join([self.dev_code, self.host_code])

    def _get_extractor(self):
        """ Returns the correct function decl extractor depending on target
            type"""
        return ISPCCFunctionDeclExtractor()

    def _visit_const(self, node):
        """Visit const arg of kernel."""
        # check the entire subdecl for ISPCUniformPointer

        pod = node
        while hasattr(pod, 'subdecl'):
            pod = pod.subdecl
            if isinstance(pod, cgen.ispc.ISPCUniformPointer):
                return self._visit_pointer(pod)

        # if not found, use POD
        self._append_arg(pod.name, pod.dtype)

    def _visit_func_decl(self, func_decl):
        """Visit nodes of function declaration of kernel."""
        for i, arg in enumerate(func_decl.arg_decls):
            if isinstance(arg, cgen.ispc.ISPCUniform):
                self._visit_const(arg)
            elif isinstance(arg, cgen.RestrictPointer):
                self._visit_pointer(arg)
            else:
                raise ValueError('unhandled type for arg %r' % (arg, ))

    def _dtype_to_ctype(self, dtype, pointer=False):
        """Map NumPy dtype to equivalent ctypes type."""
        target = self.target  # type: ISPCTarget
        registry = target.get_dtype_registry()
        typename = registry.dtype_to_ctype(dtype)
        typename = {'unsigned': 'uint'}.get(typename, typename)
        basetype = getattr(ctypes, 'c_' + typename)
        if pointer:
            return ctypes.POINTER(basetype)
        return basetype
