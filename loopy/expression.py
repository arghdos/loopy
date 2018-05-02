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

    # this is a simple list of math functions from OpenCL-1.2
    # https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mathFunctions.html
    # this could be expanded / moved to it's own target specific VecCheck if
    # necessary
    functions = """acos    acosh   acospi  asin
    asinh   asinpi  atan    atan2
    atanh   atanpi  atan2pi cbrt
    ceil    copysign    cos cosh
    cospi   erfc    erf exp
    exp2    exp10   expm1   fabs
    fdim    floor   fma fmax
    fmin    fmod    fract   frexp
    hypot   ilogb   ldexp   lgamma
    lgamma_r    log log2    log10
    log1p   logb    mad maxmag
    minmag  modf    nan nextafter
    pow pown    powr    remainder
    remquo  rint    rootn   round
    rsqrt   sin sincos  sinh
    sinpi   sqrt    tan tanh
    tanpi   tgamma  trunc"""

    functions = [x.strip() for x in functions.split() if x.strip()]

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
            if str(expr.function) not in VectorizabilityChecker.functions:
                return Unvectorizable(
                    'Function {} is not known to be vectorizable'.format(expr.name))
            return True

        return False

    @staticmethod
    def compile_time_constants(kernel, vec_iname):
        """
        Returns a dictionary of (non-vector) inames and temporary variables whose
        value is known at "compile" time. These are used (in combination with a
        codegen state's variable substitution map) to simplifying access expressions
        in :func:`get_access_info`.

        Note: inames are mapped to the :class:`Variable` version of themselves,
              while temporary variables are mapped to their integer value

        .. parameter:: kernel
            The kernel to check
        .. parameter:: vec_iname
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

        from loopy.symbolic import get_dependencies, DependencyMapper
        from loopy.kernel.array import VectorArrayDimTag

        possible = None
        for i in range(len(var.shape)):
            dep_mapper = DependencyMapper(composite_leaves=False)
            deps = dep_mapper(index[i])
            # if we're on the vector index
            if isinstance(var.dim_tags[i], VectorArrayDimTag):
                if var.shape[i] != self.vec_iname_length:
                    raise Unvectorizable("vector length was mismatched")
                if possible is None:
                    possible = self.vec_iname in [str(x) for x in deps]
            # or, if not vector index, and vector iname is present
            elif self.vec_iname in set(x.name for x in deps):
                # check whether we can simplify out the vector iname
                context = dict((x, x) for x in deps if x.name != self.vec_iname)
                allowed_symbols = self.compile_time_constants(
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
        # https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/relationalFunctions.html

        # even better for OpenCL <, <=, >, >=, !=, == are all vectorizable by default
        # (see: sec 6.3.d-6.d.3 in OpenCL-1.2 docs)

        if expr.operator in ["<", "<=", ">", ">=", "!=", "=="]:
            return any(self.rec(x) for x in [expr.left, expr.right])

        raise Unvectorizable()

    def map_logical_not(self, expr):
        # 6.3.h in OpenCL-1.2 docs
        return self.rec(expr.child)

    def map_logical_and(self, expr):
        # 6.3.h in OpenCL-1.2 docs
        return any(self.rec(x) for x in expr.children)

    map_logical_or = map_logical_and

    # sec 6.3.f in OpenCL-1.2 docs
    map_bitwise_not = map_logical_not
    map_bitwise_or = map_logical_and
    map_bitwise_xor = map_logical_and
    map_bitwise_and = map_logical_and

    def map_reduction(self, expr):
        # FIXME: Do this more carefully
        raise Unvectorizable()

# }}}

# vim: fdm=marker
