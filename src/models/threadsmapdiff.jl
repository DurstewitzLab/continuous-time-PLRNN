using ThreadsX
using Zygote, ZygoteRules, LinearAlgebra
using Zygote: _pullback, accum, _unzip, tailmemaybe
import ChainRulesCore

# Reverse utility functions for ThreadsX
_tryreverse(m, backs, Δ) = backs, Δ
_tryreverse(m::typeof(ThreadsX.map), backs, Δ) = _reverse(backs), _reverse(Δ)

_tryreverse(m, x) = x
_tryreverse(m::typeof(ThreadsX.map), x) = _reverse(x)

# Fallback reverse
_reverse(x) = reverse(x)

# Special cases for triangular/symmetric matrices
_reverse(x::LowerTriangular) = UpperTriangular(_reverse(parent(x)))
_reverse(x::UpperTriangular) = LowerTriangular(_reverse(parent(x)))
_reverse(x::UnitLowerTriangular) = UnitUpperTriangular(_reverse(parent(x)))
_reverse(x::UnitUpperTriangular) = UnitLowerTriangular(_reverse(parent(x)))
_reverse(x::Hermitian) = Hermitian(_reverse(x.data), x.uplo == 'U' ? :L : :U)
_reverse(x::Symmetric) = Symmetric(_reverse(x.data), x.uplo == 'U' ? :L : :U)

_tryaxes(x) = axes(x)
_tryaxes(x::Tuple) = Val(length(x))

_restore(dx, ax::Tuple) = axes(dx) == ax ? dx : reshape(vcat(dx, falses(prod(length, ax) - length(dx))), ax)
_restore(dx, ::Val{N}) where {N} = ntuple(i -> get(dx,i,nothing), N)

last_or_nothing(::Nothing) = nothing
last_or_nothing(x) = last(x)

# Define adjoint for ThreadsX.map
function ∇threadsxmap(cx, f::F, args::Vararg{Any, N}) where {F, N}
  ys_and_backs = ThreadsX.map((args...) -> _pullback(cx, f, args...), args...)
  ys = ThreadsX.map(first, ys_and_backs)
  arg_ax = map(_tryaxes, args)

  function map_back(Δ)
    if Base.issingletontype(F) && length(args) == 1
      Δarg = ThreadsX.map(((_,pb), δ) -> last_or_nothing(pb(δ)), ys_and_backs, Δ)
      return (nothing, Δarg)
    elseif Base.issingletontype(F)
      unzipped = _unzip(ThreadsX.map(((_,pb), δ) -> tailmemaybe(pb(δ)), ys_and_backs, Δ), Val(N))
      Δargs = map(_restore, unzipped, arg_ax)
      return (nothing, Δargs...)
    else
      Δf_and_args_zipped = ThreadsX.map(((_,pb), δ) -> pb(δ), _tryreverse(ThreadsX.map, ys_and_backs, Δ)...)
      Δf_and_args = _unzip(_tryreverse(ThreadsX.map, Δf_and_args_zipped), Val(N + 1))
      Δf = reduce(accum, Δf_and_args[1]; init=nothing)
      Δargs = map(_restore, Δf_and_args[2:end], arg_ax)
      return (Δf, Δargs...)
    end
  end

  map_back(::Nothing) = nothing
  return ys, map_back
end

# Register the adjoint with @adjoint for ThreadsX.map
@adjoint function ThreadsX.map(f, args::Union{AbstractArray,Tuple}...)
  ∇threadsxmap(__context__, f, args...)
end
