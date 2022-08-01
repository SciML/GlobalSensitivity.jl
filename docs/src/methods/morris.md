# Morris Method

```@docs
Morris(; p_steps::Array{Int, 1} = Int[], relative_scale::Bool = false,
                num_trajectory::Int = 10,
                total_num_trajectory::Int = 5 * num_trajectory, len_design_mat::Int = 10)
```

## API

```@docs
gsa(f, method::Morris, p_range::AbstractVector; batch=false, kwargs...)
```
