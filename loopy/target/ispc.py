"""Target for Intel ISPC."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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


import numpy as np  # noqa
from loopy.target.c import (CTarget, CASTBuilder, CASTIdentityMapper,
                            CFunctionDeclExtractor)
from loopy.target.c.codegen.expression import ExpressionToCExpressionMapper
from loopy.diagnostic import LoopyError
from loopy.symbolic import Literal
from pymbolic import var
import pymbolic.primitives as p
from loopy.kernel.data import temp_var_scope
from pymbolic.mapper.stringifier import PREC_NONE

from pytools import memoize_method


# {{{ expression mapper

class ExprToISPCExprMapper(ExpressionToCExpressionMapper):
    def _get_index_ctype(self):
        if self.kernel.index_dtype.numpy_dtype == np.int32:
            return "int32"
        elif self.kernel.index_dtype.numpy_dtype == np.int64:
            return "int64"
        else:
            raise ValueError("unexpected index_type")

    def map_group_hw_index(self, expr, type_context):
        return var(
                "((uniform %s) taskIndex%d)"
                % (self._get_index_ctype(), expr.axis))

    def map_local_hw_index(self, expr, type_context):
        if expr.axis == 0:
            return var("(varying %s) programIndex" % self._get_index_ctype())
        else:
            raise LoopyError("ISPC only supports one local axis")

    def map_constant(self, expr, type_context):
        if isinstance(expr, (complex, np.complexfloating)):
            raise NotImplementedError("complex numbers in ispc")
        else:
            if type_context == "f":
                return Literal(repr(float(expr)))
            elif type_context == "d":
                # Keepin' the good ideas flowin' since '66.
                return Literal(repr(float(expr))+"d")
            elif type_context == "i":
                return expr
            else:
                from loopy.tools import is_integer
                if is_integer(expr):
                    return expr

                raise RuntimeError("don't know how to generate code "
                        "for constant '%s'" % expr)

    def map_variable(self, expr, type_context):
        tv = self.kernel.temporary_variables.get(expr.name)

        if tv is not None and tv.scope == temp_var_scope.PRIVATE:
            # FIXME: This is a pretty coarse way of deciding what
            # private temporaries get duplicated. Refine? (See also
            # below in decl generation)
            gsize, lsize = self.kernel.get_grid_size_upper_bounds_as_exprs()
            if lsize:
                return expr[var("programIndex")]
            else:
                return expr

        else:
            return super(ExprToISPCExprMapper, self).map_variable(
                    expr, type_context)

    def map_subscript(self, expr, type_context):
        from loopy.kernel.data import TemporaryVariable

        ary = self.find_array(expr)

        if (isinstance(ary, TemporaryVariable)
                and ary.scope == temp_var_scope.PRIVATE):
            # generate access code for acccess to private-index temporaries

            gsize, lsize = self.kernel.get_grid_size_upper_bounds_as_exprs()
            if lsize:
                lsize, = lsize
                from loopy.kernel.array import get_access_info
                from pymbolic import evaluate

                access_info = get_access_info(self.kernel.target, ary, expr.index,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    self.codegen_state.vectorization_info)

                subscript, = access_info.subscripts
                result = var(access_info.array_name)[
                        var("programIndex") + self.rec(lsize*subscript, 'i')]

                if access_info.vector_index is not None:
                    return self.kernel.target.add_vector_access(
                        result, access_info.vector_index)
                else:
                    return result

        return super(ExprToISPCExprMapper, self).map_subscript(
                expr, type_context)

# }}}


# {{{ type registry

def fill_registry_with_ispc_types(reg, respect_windows, include_bool=True):
    reg.get_or_register_dtype("bool", np.bool)

    reg.get_or_register_dtype(["int8", "signed char", "char"], np.int8)
    reg.get_or_register_dtype(["uint8", "unsigned char"], np.uint8)
    reg.get_or_register_dtype(["int16", "short", "signed short",
        "signed short int", "short signed int"], np.int16)
    reg.get_or_register_dtype(["uint16", "unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    reg.get_or_register_dtype(["int32", "int", "signed int"], np.int32)
    reg.get_or_register_dtype(["uint32", "unsigned", "unsigned int"], np.uint32)

    reg.get_or_register_dtype(["int64"], np.int64)
    reg.get_or_register_dtype(["uint64"], np.uint64)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)

# }}}


class ISPCTarget(CTarget):
    """A code generation target for Intel's `ISPC <https://ispc.github.io/>`_
    SPMD programming language, to target Intel's Knight's hardware and modern
    Intel CPUs with wide vector units.
    """

    def __init__(self, occa_mode=False, compiler=None):
        """
        :arg occa_mode: Whether to modify the generated call signature to
            be compatible with OCCA
        :compiler: An instance of :class:`ISPCCompiler` or similar that controls
            compilation for this target
        """
        self.occa_mode = occa_mode
        self.compiler = compiler

        super(ISPCTarget, self).__init__()

    host_program_name_suffix = ""
    device_program_name_suffix = "_inner"

    def pre_codegen_check(self, kernel):
        gsize, lsize = kernel.get_grid_size_upper_bounds_as_exprs()
        if len(lsize) > 1:
            for i, ls_i in enumerate(lsize[1:]):
                if ls_i != 1:
                    raise LoopyError("local axis %d (0-based) "
                            "has length > 1, which is unsupported "
                            "by ISPC" % ls_i)

    def get_host_ast_builder(self):
        return ISPCASTBuilder(self)

    def get_device_ast_builder(self):
        return ISPCASTBuilder(self)

    def get_kernel_executor_cache_key(self, *args, **kwargs):
        return None

    def get_kernel_executor(self, knl, *args, **kwargs):
        from loopy.target.ispc_execution import ISPCKernelExecutor
        return ISPCKernelExecutor(knl, compiler=self.compiler)

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import DTypeRegistry
        result = DTypeRegistry()
        fill_registry_with_ispc_types(result, respect_windows=False,
                include_bool=True)
        return result

    # }}}


class ISPCASTBuilder(CASTBuilder):
    def _arg_names_and_decls(self, codegen_state):
        implemented_data_info = codegen_state.implemented_data_info
        arg_names = [iai.name for iai in implemented_data_info]

        arg_decls = [
                self.idi_to_cgen_declarator(codegen_state.kernel, idi)
                for idi in implemented_data_info]

        # {{{ occa compatibility hackery

        from cgen import Value
        if self.target.occa_mode:
            from cgen import ArrayOf, Const
            from cgen.ispc import ISPCUniform

            arg_decls = [
                    Const(ISPCUniform(ArrayOf(Value("int", "loopy_dims")))),
                    Const(ISPCUniform(Value("int", "o1"))),
                    Const(ISPCUniform(Value("int", "o2"))),
                    Const(ISPCUniform(Value("int", "o3"))),
                    ] + arg_decls
            arg_names = ["loopy_dims", "o1", "o2", "o3"] + arg_names

        # }}}

        return arg_names, arg_decls

    # {{{ top-level codegen

    def get_function_declaration(self, codegen_state, codegen_result,
            schedule_index):
        name = codegen_result.current_program(codegen_state).name

        from cgen import (FunctionDeclaration, Value)
        from cgen.ispc import ISPCExport, ISPCTask

        arg_names, arg_decls = self._arg_names_and_decls(codegen_state)

        if codegen_state.is_generating_device_code:
            result = ISPCTask(
                        FunctionDeclaration(
                            Value("void", name),
                            arg_decls))
        else:
            result = ISPCExport(
                    FunctionDeclaration(
                        Value("void", name),
                        arg_decls))

        from loopy.target.c import FunctionDeclarationWrapper
        return FunctionDeclarationWrapper(result)

    # }}}

    def get_kernel_call(self, codegen_state, name, gsize, lsize, extra_args):
        ecm = self.get_expression_to_code_mapper(codegen_state)

        from pymbolic.mapper.stringifier import PREC_NONE
        result = []
        from cgen import Statement as S, Block
        if lsize:
            result.append(
                    S(
                        "assert(programCount == (%s))"
                        % ecm(lsize[0], PREC_NONE)))

        arg_names, arg_decls = self._arg_names_and_decls(codegen_state)

        from cgen.ispc import ISPCLaunch
        result.append(
                ISPCLaunch(
                    tuple(ecm(gs_i, PREC_NONE) for gs_i in gsize),
                    "%s(%s)" % (
                        name,
                        ", ".join(arg_names)
                        )))

        return Block(result)

    # {{{ code generation guts

    def get_expression_to_c_expression_mapper(self, codegen_state):
        return ExprToISPCExprMapper(codegen_state)

    def add_vector_access(self, access_expr, index):
        return access_expr[index]

    def emit_barrier(self, kind, comment):
        from cgen import Comment, Statement

        assert comment

        if kind == "local":
            return Comment("local barrier: %s" % comment)

        elif kind == "global":
            return Statement("sync; /* %s */" % comment)

        else:
            raise LoopyError("unknown barrier kind")

    def get_temporary_decl(self, codegen_state, sched_index, temp_var, decl_info):
        from loopy.target.c import POD  # uses the correct complex type
        temp_var_decl = POD(self, decl_info.dtype, decl_info.name)

        shape = decl_info.shape

        if temp_var.scope == temp_var_scope.PRIVATE:
            # FIXME: This is a pretty coarse way of deciding what
            # private temporaries get duplicated. Refine? (See also
            # above in expr to code mapper)
            _, lsize = codegen_state.kernel.get_grid_size_upper_bounds_as_exprs()
            shape = lsize + shape

        if shape:
            from cgen import ArrayOf
            ecm = self.get_expression_to_code_mapper(codegen_state)
            temp_var_decl = ArrayOf(
                    temp_var_decl,
                    ecm(p.flattened_product(shape),
                        prec=PREC_NONE, type_context="i"))

        return temp_var_decl

    def wrap_temporary_decl(self, decl, scope):
        from cgen.ispc import ISPCUniform
        return ISPCUniform(decl)

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from loopy.target.c import POD  # uses the correct complex type
        from cgen import Const
        from cgen.ispc import ISPCUniformPointer, ISPCUniform

        arg_decl = ISPCUniformPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        arg_decl = ISPCUniform(arg_decl)

        return arg_decl

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        result = super(ISPCASTBuilder, self).get_value_arg_decl(
                name, shape, dtype, is_written)

        from cgen import Reference, Const
        was_const = isinstance(result, Const)

        if was_const:
            result = result.subdecl

        if self.target.occa_mode:
            result = Reference(result)

        if was_const:
            result = Const(result)

        from cgen.ispc import ISPCUniform
        return ISPCUniform(result)

    def emit_assignment(self, codegen_state, insn):
        kernel = codegen_state.kernel
        ecm = codegen_state.expression_to_code_mapper

        assignee_var_name, = insn.assignee_var_names()

        lhs_var = codegen_state.kernel.get_var_descriptor(assignee_var_name)
        lhs_dtype = lhs_var.dtype

        if insn.atomicity is not None:
            lhs_atomicity = [
                    a for a in insn.atomicity if a.var_name == assignee_var_name]
            assert len(lhs_atomicity) <= 1
            if lhs_atomicity:
                lhs_atomicity, = lhs_atomicity
            else:
                lhs_atomicity = None
        else:
            lhs_atomicity = None

        from loopy.kernel.data import AtomicInit, AtomicUpdate
        from loopy.expression import dtype_to_type_context

        from pymbolic.mapper.stringifier import PREC_NONE
        lhs_code = ecm(insn.assignee, prec=PREC_NONE, type_context=None)
        rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
        if lhs_atomicity is None:
            from cgen import Assign
            return Assign(
                    lhs_code,
                    ecm(insn.expression, prec=PREC_NONE,
                        type_context=rhs_type_context,
                        needed_dtype=lhs_dtype))

        elif isinstance(lhs_atomicity, AtomicInit):
            codegen_state.seen_atomic_dtypes.add(lhs_dtype)
            return codegen_state.ast_builder.emit_atomic_init(
                    codegen_state, lhs_atomicity, lhs_var,
                    insn.assignee, insn.expression,
                    lhs_dtype, rhs_type_context)

        elif isinstance(lhs_atomicity, AtomicUpdate):
            codegen_state.seen_atomic_dtypes.add(lhs_dtype)
            return codegen_state.ast_builder.emit_atomic_update(
                    codegen_state, lhs_atomicity, lhs_var,
                    insn.assignee, insn.expression,
                    lhs_dtype, rhs_type_context)

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE

        rhs_type_context = dtype_to_type_context(kernel.target, lhs_dtype)
        rhs_code = ecm(insn.expression, prec=PREC_NONE,
                    type_context=rhs_type_context,
                    needed_dtype=lhs_dtype)

        lhs = insn.assignee

        # {{{ handle streaming stores

        if "!streaming_store" in insn.tags:
            ary = ecm.find_array(lhs)

            from loopy.kernel.array import get_access_info
            from pymbolic import evaluate

            from loopy.symbolic import simplify_using_aff
            index_tuple = tuple(
                    simplify_using_aff(kernel, idx) for idx in lhs.index_tuple)

            access_info = get_access_info(kernel.target, ary, index_tuple,
                    lambda expr: evaluate(expr, self.codegen_state.var_subst_map),
                    codegen_state.vectorization_info)

            from loopy.kernel.data import GlobalArg, TemporaryVariable

            if not isinstance(ary, (GlobalArg, TemporaryVariable)):
                raise LoopyError("array type not supported in ISPC: %s"
                        % type(ary).__name)

            if len(access_info.subscripts) != 1:
                raise LoopyError("streaming stores must have a subscript")
            subscript, = access_info.subscripts

            from pymbolic.primitives import Sum, flattened_sum, Variable
            if isinstance(subscript, Sum):
                terms = subscript.children
            else:
                terms = (subscript.children,)

            new_terms = []

            from loopy.kernel.data import LocalIndexTag
            from loopy.symbolic import get_dependencies

            saw_l0 = False
            for term in terms:
                if (isinstance(term, Variable)
                        and isinstance(
                            kernel.iname_to_tag.get(term.name), LocalIndexTag)
                        and kernel.iname_to_tag.get(term.name).axis == 0):
                    if saw_l0:
                        raise LoopyError("streaming store must have stride 1 "
                                "in local index, got: %s" % subscript)
                    saw_l0 = True
                    continue
                else:
                    for dep in get_dependencies(term):
                        if (
                                isinstance(
                                    kernel.iname_to_tag.get(dep), LocalIndexTag)
                                and kernel.iname_to_tag.get(dep).axis == 0):
                            raise LoopyError("streaming store must have stride 1 "
                                    "in local index, got: %s" % subscript)

                    new_terms.append(term)

            if not saw_l0:
                raise LoopyError("streaming store must have stride 1 in "
                        "local index, got: %s" % subscript)

            if access_info.vector_index is not None:
                raise LoopyError("streaming store may not use a short-vector "
                        "data type")

            rhs_has_programindex = any(
                    isinstance(
                        kernel.iname_to_tag.get(dep), LocalIndexTag)
                    and kernel.iname_to_tag.get(dep).axis == 0
                    for dep in get_dependencies(insn.expression))

            if not rhs_has_programindex:
                rhs_code = "broadcast(%s, 0)" % rhs_code

            from cgen import Statement
            return Statement(
                    "streaming_store(%s + %s, %s)"
                    % (
                        access_info.array_name,
                        ecm(flattened_sum(new_terms), PREC_NONE, 'i'),
                        rhs_code))

        # }}}

        from cgen import Assign
        return Assign(ecm(lhs, prec=PREC_NONE, type_context=None), rhs_code)

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            lbound, ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.target.c import POD

        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For, InlineInitializer

        from cgen.ispc import ISPCUniform

        return For(
                InlineInitializer(
                    ISPCUniform(POD(self, iname_dtype, iname)),
                    ecm(lbound, PREC_NONE, "i")),
                ecm(
                    p.Comparison(var(iname), "<=", ubound),
                    PREC_NONE, "i"),
                "++%s" % iname,
                inner)
    # }}}

    def _atomic_is_global(self, codegen_state, lhs_expr, rhs_expr):
        # there must be a better way to do this, but for the moment we compare
        # (stringwise) any indicies that involve a global tag, and ensure that
        # they are identical on the left and right hand sides
        # if they are, this is a local atomic
        from six import iteritems
        from loopy.kernel.data import GroupIndexTag
        import re
        gtags = set(re.compile(r'\b' + name + r'\b')
            for name, tag in iteritems(
            codegen_state.kernel.iname_to_tag)
                    if isinstance(tag, GroupIndexTag))

        var_kind = '_local'
        index_extractor = re.compile(r'\[([^\]]+)\]')

        # find the LHS indexing strings
        lhs_ind = index_extractor.search(str(lhs_expr)).groups()
        if lhs_ind:
            lhs_ind = lhs_ind[0]
            # split the LHS by comma's if possible
            lhs_ind = [x.strip() for x in lhs_ind.split(',') if x.strip()]
            # now filter to find only those that have a global tag
            lhs_ind = [x for x in lhs_ind if any(tag.search(x) for tag in gtags)]

            # repeat for the RHS indexs
            rhs_inds = index_extractor.findall(str(rhs_expr))
            rhs_inds = [x.strip() for y in rhs_inds for x in y.split(',')
                        if x.strip()]
            # filter
            rhs_inds = [x for x in rhs_inds if any(tag.search(x) for tag in gtags)]

            # find any that are not on lhs
            if [x for x in rhs_inds if x not in lhs_ind]:
                var_kind = '_global'
                # Once you get to global atomics, several common parameter
                # patterns simply don't exist.  It will require some thought
                # on how best to handle it.

                # For the moment, it's safer to just raise an exception
                raise NotImplementedError('Global atomics are not fully'
                    ' implemented for ISPC.')

        return var_kind

    # {{{ code generation for emitting a generatic atomic operations

    def _emit_atomic_operation(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context, func_name,
            need_loop=True):
        from pymbolic.mapper.stringifier import PREC_NONE
        from loopy.types import NumpyType

        kind = self._atomic_is_global(codegen_state, lhs_expr, rhs_expr)
        func_name = '%(func_name)s%(kind)s' % {
            'func_name': func_name,
            'kind': kind
        }

        if isinstance(lhs_dtype, NumpyType) and lhs_dtype.numpy_dtype in [
                np.int32, np.int64, np.float32, np.float64]:
            from cgen import Block, DoWhile, Assign, Statement
            from loopy.target.c import POD
            from pymbolic.mapper.substitutor import make_subst_func
            from pymbolic import var
            from loopy.symbolic import SubstitutionMapper

            if not need_loop:
                # first, get a copy of the rhs_expr with the lhs_expr removed
                # (if there)
                ecm = codegen_state.expression_to_code_mapper

                # get lhs
                lhs_expr_code = ecm(lhs_expr, prec=PREC_NONE, type_context=None)

                # sub 0 for lhs in rhs
                subst = SubstitutionMapper(
                        make_subst_func({lhs_expr: 0}))
                rhs_expr_code = ecm(subst(rhs_expr), prec=PREC_NONE,
                        type_context=rhs_type_context,
                        needed_dtype=lhs_dtype)

                return Statement(
                    "%(func_name)s("
                    "&%(lhs_expr)s, "
                    "%(rhs_expr)s)" % {
                        'func_name': func_name,
                        'lhs_expr': lhs_expr_code,
                        'rhs_expr': rhs_expr_code,
                    })

            # otherwise, we need to do the general loop form of this
            old_val_var = codegen_state.var_name_generator("loopy_old_val")
            new_val_var = codegen_state.var_name_generator("loopy_new_val")

            from loopy.kernel.data import TemporaryVariable
            ecm = codegen_state.expression_to_code_mapper.with_assignments(
                    {
                        old_val_var: TemporaryVariable(old_val_var, lhs_dtype),
                        new_val_var: TemporaryVariable(new_val_var, lhs_dtype),
                        })

            lhs_expr_code = ecm(lhs_expr, prec=PREC_NONE, type_context=None)

            subst = SubstitutionMapper(
                    make_subst_func({lhs_expr: var(old_val_var)}))
            rhs_expr_code = ecm(subst(rhs_expr), prec=PREC_NONE,
                    type_context=rhs_type_context,
                    needed_dtype=lhs_dtype)

            return Block([
                POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                    old_val_var),
                POD(self, NumpyType(lhs_dtype.dtype, target=self.target),
                    new_val_var),
                DoWhile(
                    "%(func_name)s("
                    "&%(lhs_expr)s, "
                    "%(old_val)s, "
                    "%(new_val)s"
                    ") != %(old_val)s"
                    % {
                        "func_name": func_name,
                        "lhs_expr": lhs_expr_code,
                        "old_val": old_val_var,
                        "new_val": new_val_var,
                        },
                    Block([
                        Assign(old_val_var, lhs_expr_code),
                        Assign(new_val_var, rhs_expr_code),
                        ])
                    )
                ])
        else:
            raise NotImplementedError("atomic update for '%s'" % lhs_dtype)

    # }}}

    # {{{ code generation for atomic init

    def emit_atomic_init(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):
        return self._emit_atomic_operation(codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context,
            'atomic_compare_exchange', need_loop=True)

    # }}}

    def _get_atomic_func_name(self, lhs_expr, rhs_expr):
        # first, we search for the lhs_expr in the rhs_expr
        from pymbolic.primitives import Sum, LogicalAnd, LogicalOr
        try:
            # if it's a simple product / sum / quotient, and we can find
            # LHS in the children
            contains = any(x == lhs_expr for x in rhs_expr.children)
        except AttributeError:
            # otherwise, we need a complex atomic
            return None

        # not a simple atomic
        if not contains:
            return None

        if isinstance(rhs_expr, Sum):
            return 'atomic_add'
        elif isinstance(rhs_expr, LogicalAnd):
            return 'atomic_and'
        elif isinstance(rhs_expr, LogicalOr):
            return 'atomic_or'

        # not found
        return None

    # {{{ code generation for atomic update

    def emit_atomic_update(self, codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context):
        # find appropriate function name if possible
        func_name = self._get_atomic_func_name(lhs_expr, rhs_expr)
        need_loop = False
        if func_name is None:
            func_name = 'atomic_compare_exchange'
            need_loop = True

        return self._emit_atomic_operation(codegen_state, lhs_atomicity, lhs_var,
            lhs_expr, rhs_expr, lhs_dtype, rhs_type_context, func_name=func_name,
            need_loop=need_loop)

    # }}}


class ISPCCFunctionDeclExtractor(CASTIdentityMapper):
    def __init__(self):
        self.decls = []

    def map_expression(self, expr):
        return expr

    def map_function_decl_wrapper(self, node):
        self.decls.append(node.subdecl.subdecl)
        return super(ISPCCFunctionDeclExtractor, self)\
                .map_function_decl_wrapper(node)


# TODO: Generate launch code
# TODO: Vector types (element access: done)

# vim: foldmethod=marker
