from __future__ import division, absolute_import, print_function

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

import numpy as np
import loopy as lp
import pytest
from loopy.target.ispc import ISPCTarget

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def test_ispc_target():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        [
            lp.GlobalArg("out", np.float32, shape=lp.auto),
            lp.GlobalArg("a", np.float32, shape=lp.auto),
            "..."
        ],
        target=ISPCTarget())

    assert np.allclose(knl(a=np.arange(16, dtype=np.float32))[1],
                       2 * np.arange(16, dtype=np.float32))


def test_ispc_target_strides():
    from loopy.target.ispc import ISPCTarget

    def __get_kernel(order='C'):
        return lp.make_kernel(
            "{ [i,j]: 0<=i,j<n }",
            "out[i, j] = 2*a[i, j]",
            [
                lp.GlobalArg(
                    "out", np.float32, shape=('n', 'n'), order=order),
                lp.GlobalArg(
                    "a", np.float32, shape=('n', 'n'), order=order),
                "..."
            ],
            target=ISPCTarget())

    # test with C-order
    knl = __get_kernel('C')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='C')

    assert np.allclose(knl(a=a_np)[1],
                       2 * a_np)

    # test with F-order
    knl = __get_kernel('F')
    a_np = np.reshape(np.arange(16 * 16, dtype=np.float32), (16, -1),
                      order='F')

    assert np.allclose(knl(a=a_np)[1],
                       2 * a_np)


def test_ispc_target_strides_nonsquare():
    def __get_kernel(order='C'):
        indicies = ['i', 'j', 'k']
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = '{{ [{iname}]: 0 <= {iname} < {size} }}'
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = 'out[{indexed}] = 2 * a[{indexed}]'.format(
            indexed=', '.join(indicies))
        return lp.make_kernel(
            domains,
            statement,
            [
                lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                "..."
            ],
            target=ISPCTarget())

    # test with C-order
    knl = __get_kernel('C')
    a_lp = next(x for x in knl.args if x.name == 'a')
    a_np = np.reshape(np.arange(np.product(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order='C')

    assert np.allclose(knl(a=a_np)[1],
                       2 * a_np)

    # test with F-order
    knl = __get_kernel('F')
    a_lp = next(x for x in knl.args if x.name == 'a')
    a_np = np.reshape(np.arange(np.product(a_lp.shape), dtype=np.float32),
                      a_lp.shape,
                      order='F')

    assert np.allclose(knl(a=a_np)[1],
                       2 * a_np)


def test_ispc_optimizations():
    def __get_kernel(order='C'):
        indicies = ['i', 'j', 'k']
        sizes = tuple(np.random.randint(1, 11, size=len(indicies)))
        # create domain strings
        domain_template = '{{ [{iname}]: 0 <= {iname} < {size} }}'
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = 'out[{indexed}] = 2 * a[{indexed}]'.format(
            indexed=', '.join(indicies))
        return lp.make_kernel(
            domains,
            statement,
            [
                lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                "..."
            ],
            target=ISPCTarget()), sizes

    # test with ILP
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='ilp')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)

    # test with unrolling
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='unr')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)

    # test with vectorization
    knl, sizes = __get_kernel('C')
    knl = lp.split_iname(knl, 'i', 4, inner_tag='l.0', outer_tag='g.0')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    assert np.allclose(knl(a=a_np)[1], 2 * a_np)


def test_bad_vecsize_fails():
    def __get_kernel(order='C'):
        indicies = ['i', 'j', 'k']
        sizes = tuple(np.random.randint(4, 11, size=len(indicies)))
        # create domain strings
        domain_template = '{{ [{iname}]: 0 <= {iname} < {size} }}'
        domains = []
        for idx, size in zip(indicies, sizes):
            domains.append(domain_template.format(
                iname=idx,
                size=size))
        statement = 'out[{indexed}] = 2 * a[{indexed}]'.format(
            indexed=', '.join(indicies))
        return lp.make_kernel(
            domains,
            statement,
            [
                lp.GlobalArg("out", np.float32, shape=sizes, order=order),
                lp.GlobalArg("a", np.float32, shape=sizes, order=order),
                "..."
            ],
            target=ISPCTarget()), sizes

    knl, sizes = __get_kernel('C')
    # can't vectorize w/ size 3
    knl = lp.split_iname(knl, 'i', 3, inner_tag='l.0', outer_tag='g.0')
    a_np = np.reshape(np.arange(np.product(sizes), dtype=np.float32),
                      sizes,
                      order='C')

    from codepy.toolchain import CompileError
    with pytest.raises(CompileError):
        assert np.allclose(knl(a=a_np)[1], 2 * a_np)


@pytest.mark.parametrize('vec_width', [4, 8, 16])
@pytest.mark.parametrize('target', ['sse2', 'sse4', 'avx1', 'av2'])
@pytest.mark.parametrize('n', [10, 100])
def test_ispc_vector_sizes_and_targets(vec_width, target, n):
    from loopy.target.ispc_execution import ISPCCompiler

    if target == 'sse4' and vec_width == 16:
        pytest.skip('Not recognized by ispc')

    compiler = ISPCCompiler(vector_width=vec_width, target=target)
    target = ISPCTarget(compiler=compiler)

    knl = lp.make_kernel(
        '{[i]: 0<=i<n}',
        """
            out[i] = 2 * a[i]
            """,
        [lp.GlobalArg("a", shape=(n,)),
         lp.GlobalArg("out", shape=(n,))],
        target=target)

    knl = lp.fix_parameters(knl, n=n)

    a_np = np.arange(n)
    from loopy import LoopyError
    try:
        assert np.allclose(knl(a=a_np)[1], 2 * a_np)
    except LoopyError:
        assert n == 10
        pytest.xfail('Issue from mailing list')


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("atomic_type", ['l.0', 'g.0'])
def test_atomic(dtype, atomic_type):
    m = 4
    n = 10000
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        """
            <> ind = indexer[i]
            out[ind] = out[ind] + 2 * a[i] {atomic}
        """,
        [
            lp.GlobalArg("out", dtype, shape=(m,), for_atomic=True),
            lp.GlobalArg("a", dtype, shape=(n,)),
            lp.GlobalArg("indexer", dtype=np.int32, shape=(n,))
        ],
        target=ISPCTarget(),
        assumptions="n>0")

    # create base array
    a = np.arange(n, dtype=dtype)
    indexer = (a % m).astype(dtype=np.int32)
    out = np.zeros((m,), dtype=dtype)
    for i in range(n):
        out[indexer[i]] += 2 * a[i]

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.split_iname(knl, "i", 8, inner_tag=atomic_type)
    try:
        _, test = knl(a=a, out=np.zeros_like(out), indexer=indexer.copy())
        assert np.allclose(test[0], out)
    except NotImplementedError:
        assert atomic_type == 'g.0'
        pytest.xfail('Global atomics not implemented')


def test_atomic_init():
    dtype = np.int32
    atomic_type = 'l.0'
    m = 4
    n = 10000
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        """
            <>ind = indexer[i]
            out[ind] = i {id=init, atomic=init}
        """,
        [
            lp.GlobalArg("out", dtype, shape=(m,), for_atomic=True),
            lp.GlobalArg("indexer", dtype=np.int32, shape=(n,))
        ],
        target=ISPCTarget(),
        assumptions="n>0")

    # create base array
    a = np.arange(n, dtype=dtype)
    indexer = (a % m).astype(dtype=np.int32)
    out = np.zeros((m,), dtype=dtype)

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.split_iname(knl, "i", 8, inner_tag=atomic_type)
    knl = knl.copy(silenced_warnings=['write_race(init)'])

    _, test = knl(out=np.zeros_like(out), indexer=indexer.copy())
    assert np.allclose(indexer[test], np.arange(m))


@pytest.mark.parametrize('atomic_type', ['varying', 'uniform'])
def test_uniform_atomics(atomic_type):
    dtype = np.int32
    m = 4
    n = 10000
    knl = lp.make_kernel(
        "{ [i,j]: 0<=i,j<n }",
        """
        for j
            <> upper = n
            <> lower = 0
            for i
                upper = upper - i {id=sum0}
                lower = lower + i * i {id=sum1}
            end
            out[%(index)s] = out[%(index)s] + \
                %(val)s {id=set, dep=sum*, atomic%(atype)s}
        end
        """ % {'atype': '' if atomic_type == 'varying' else '=uniform',
               'index': 'indexer[j]' if atomic_type == 'varying' else '0',
               'val': 'upper / lower' if atomic_type == 'varying' else '1'},
        [
            lp.GlobalArg("out", dtype=dtype, shape=(m,), for_atomic=True),
            lp.GlobalArg("indexer", dtype=dtype, shape=(n,))
        ],
        target=ISPCTarget(),
        assumptions="n>0")

    # create base array
    a = np.arange(n, dtype=dtype)
    indexer = (a % m).astype(dtype=np.int32)
    out = np.zeros((m,), dtype=dtype)

    knl = lp.fix_parameters(knl, n=n)
    knl = lp.split_iname(knl, "j", 8, inner_tag='l.0')
    knl = knl.copy(silenced_warnings=['write_race(set)'])

    _, test = knl(out=np.zeros_like(out), indexer=indexer.copy())
    # get the answer
    # each i sum
    i_sum = 6 * (n - 0.5 * n * (n + 1)) / (n * (n + 1) * (2 * n + 1))
    ref = np.zeros_like(out)
    if atomic_type == 'varying':
        # for the varying type, each j will add to the appropriate out
        for i in range(m):
            ref[i] = np.where(indexer == i)[0].size * i_sum
    else:
        # for the uniform atomic, the program will only perform one atomic update
        # hence
        ref[0] = np.ceil(n / 8.0)

    assert np.allclose(test, ref)


def test_temporary_duplication_remover():
    dtype = np.int32
    m = 10
    from loopy.kernel.data import temp_var_scope
    knl = lp.make_kernel(
        "{ [i]: 0<=i<10 }",
        """
        out[i] = 2.0 * a[i]
        """,
        [
            lp.GlobalArg("out", dtype, shape=(m,)),
            lp.TemporaryVariable("a", dtype=dtype, shape=(m,),
                                 initializer=np.arange(m, dtype=dtype),
                                 read_only=True, scope=temp_var_scope.GLOBAL)
        ],
        target=ISPCTarget())

    knl = lp.split_iname(knl, 'i', 8, inner_tag='l.0')
    _, test = knl()
    assert np.allclose(test, 2 * np.arange(m))
