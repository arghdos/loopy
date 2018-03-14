from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2012-15 Andreas Kloeckner"

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


import numpy as np

from pymbolic.mapper import RecursiveMapper

from loopy.codegen import Unvectorizable
from loopy.diagnostic import LoopyError


# type_context may be:
# - 'i' for integer -
# - 'f' for single-precision floating point
# - 'd' for double-precision floating point
# or None for 'no known context'.

def dtype_to_type_context(target, dtype):
    from loopy.types import NumpyType

    if dtype.is_integral():
        return 'i'
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float64, np.complex128]:
        return 'd'
    if isinstance(dtype, NumpyType) and dtype.dtype in [np.float32, np.complex64]:
        return 'f'
    if target.is_vector_dtype(dtype):
        return dtype_to_type_context(
                target, NumpyType(dtype.numpy_dtype.fields["x"][0]))

    return None


# {{{ vetorizability checker

class VectorizabilityChecker(RecursiveMapper):
    """The return value from this mapper is a :class:`bool` indicating whether
    the result of the expression is vectorized along :attr:`vec_iname`.
    If the expression is not vectorizable, the mapper raises :class:`Unvectorizable`.

    .. attribute:: vec_iname
    """

    def __init__(self, kernel, vec_iname, vec_iname_length):
        self.kernel = kernel
        self.vec_iname = vec_iname
        self.vec_iname_length = vec_iname_length

    @staticmethod
    def combine(vectorizabilities):
        from functools import reduce
        from operator import and_
        return reduce(and_, vectorizabilities)

    def map_sum(self, expr):
        return any([self.rec(child) for child in expr.children])

    map_product = map_sum

    def map_quotient(self, expr):
        return (self.rec(expr.numerator)
                or
                self.rec(expr.denominator))

    def map_linear_subscript(self, expr):
        return False

    def map_call(self, expr):
        # FIXME: Should implement better vectorization check for function calls

        rec_pars = [
                self.rec(child) for child in expr.parameters]
        if any(rec_pars):
            raise Unvectorizable("fucntion calls cannot yet be vectorized")

        return False

    @staticmethod
    def allowed_non_vecdim_dependencies(kernel, vec_iname):
        """
        Returns the dictionary of non-vector inames and compile time constants
        mapped to their 'value' (themselves in case of iname, integer value in case
        of constant)

        .. attribute:: kernel
            The kernel to check
        .. attribute:: vec_iname
            the vector iname
        """

        # determine allowed symbols as non-vector inames
        from pymbolic.primitives import Variable
        allowed_symbols = dict((sym, Variable(sym)) for sym in kernel.iname_to_tag
                               if sym != vec_iname)
        from loopy.kernel.instruction import Assignment
        from loopy.tools import is_integer
        from six import iteritems

        # and compile time integer temporaries
        compile_time_assign = dict((str(insn.assignee), insn.expression)
                                   for insn in kernel.instructions if
                                   isinstance(insn, Assignment) and is_integer(
                                   insn.expression))
        allowed_symbols.update(
            dict((sym, compile_time_assign[sym]) for sym, var in iteritems(
                    kernel.temporary_variables)
                # temporary variables w/ no initializer, no shape
                if var.initializer is None and not var.shape
                # compile time integers
                and sym in compile_time_assign))
        return allowed_symbols

    def map_subscript(self, expr):
        name = expr.aggregate.name

        var = self.kernel.arg_dict.get(name)
        if var is None:
            var = self.kernel.temporary_variables.get(name)

        if var is None:
            raise LoopyError("unknown array variable in subscript: %s"
                    % name)

        from loopy.kernel.array import ArrayBase
        if not isinstance(var, ArrayBase):
            raise LoopyError("non-array subscript '%s'" % expr)

        index = expr.index_tuple

        from loopy.symbolic import get_dependencies
        from loopy.kernel.array import VectorArrayDimTag
        from pymbolic.primitives import Variable

        possible = None
        for i in range(len(var.shape)):
            # if index is exactly vector iname
            if isinstance(var.dim_tags[i], VectorArrayDimTag) and (
                    (isinstance(index[i], Variable)
                     and index[i].name == self.vec_iname)):
                if var.shape[i] != self.vec_iname_length:
                    raise Unvectorizable("vector length was mismatched")

                if possible is None:
                    possible = True

            # or, if not vector index, and vector iname is present
            elif not isinstance(var.dim_tags[i], VectorArrayDimTag):
                from loopy.symbolic import DependencyMapper
                dep_mapper = DependencyMapper(composite_leaves=False)
                deps = dep_mapper(index[i])
                if self.vec_iname in set(x.name for x in deps):
                    # check whether we can simplify out the vector iname
                    context = dict((x, x) for x in deps if x.name != self.vec_iname)
                    allowed_symbols = self.allowed_non_vecdim_dependencies(
                        self.kernel, self.vec_iname)

                    from pymbolic import substitute
                    from pymbolic.mapper.evaluator import UnknownVariableError
                    from loopy.tools import is_integer
                    for veci in range(self.vec_iname_length):
                        ncontext = context.copy()
                        ncontext[self.vec_iname] = veci
                        try:
                            idi = substitute(index[i], ncontext)
                            if not is_integer(idi) and not all(
                                    x in allowed_symbols
                                    for x in get_dependencies(idi)):
                                raise Unvectorizable(
                                    "vectorizing iname '%s' occurs in "
                                    "unvectorized subscript axis %d (1-based) of "
                                    "expression '%s', and could not be simplified"
                                    "to compile-time constants."
                                    % (self.vec_iname, i+1, expr))
                        except UnknownVariableError:
                            break

        return bool(possible)

    def map_constant(self, expr):
        return False

    def map_variable(self, expr):
        if expr.name == self.vec_iname:
            # Technically, this is doable. But we're not going there.
            raise Unvectorizable()

        # A single variable is always a scalar.
        return False

    map_tagged_variable = map_variable

    def map_lookup(self, expr):
        if self.rec(expr.aggregate):
            raise Unvectorizable()

        return False

    def map_comparison(self, expr):
        # FIXME: These actually can be vectorized:
        # https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/relationalFunctions.html

        raise Unvectorizable()

    def map_logical_not(self, expr):
        raise Unvectorizable()

    map_logical_and = map_logical_not
    map_logical_or = map_logical_not

    def map_reduction(self, expr):
        # FIXME: Do this more carefully
        raise Unvectorizable()

# }}}

# vim: fdm=marker
