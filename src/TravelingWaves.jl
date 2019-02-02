module Waves

using Optim
using JLD2
using Plots
using Dates

# From Biswas 2009, Section 3: Example
# p[1]: A0, p[2]: a, p[3]: n, p[4]: c
@. v(t,p) = (2 * exp(-p[3] * p[2] * t)) / (p[2] * p[3] * t * (p[3] + 1) * (p[3] + 2 ))
@. A(t,p) = p[1] * exp(-p[2] * t)
@. B(p) = p[3] / sqrt( 2 * p[4] * (p[3] + 1) * (p[3] + 2))
decaying_sech2(x,p) = A(x[:,end],p) ./ (cosh.(B(p) .* (x[:,1:end-1] .- v(x[:,end],p) .* x[:,end]))).^(2 ./ p[3])

@. decaying_linear_ramp(x,p) = p[1] * abs(x[:,1:end-1] - p[2] * x[:,end])
@. decaying_exponential_ramp(x,p) = p[1] * exp(-p[2] * abs(x[:,1:end-1] - p[3] * x[:,end]))

ramped_decaying_sech2(x,p) = decaying_sech2(x,p[1:end-2]) + decaying_linear_ramp(x,p[end-1:end])
exponentially_ramped_decaying_sech2(x,p) = decaying_sech2(x,p[1:4]) + decaying_exponential_ramp(x,p[5:7])
two_sechs(x,p) = decaying_sech2(x,p[1:4]) + decaying_sech2(x,p[5:8])


function get_wave_data(file_name)
	@load file_name wave x t
	return wave, x, t
end

function construct_xdata(x,t)
	xdata = repeat([x...,0.0]', outer=(length(t),))
	xdata[:,end] .= t
	return xdata
end

function get_sumsq_loss_fn(fn, wave, xdata)
	sumsq_loss(p) = sum((fn(xdata, p) .- wave').^2)
	return sumsq_loss
end

function fit_data(wave, xdata, fn, p0, lower, upper, alg, autodiff)
	sumsq_loss = get_sumsq_loss_fn(fn, wave, xdata)
	result = optimize(sumsq_loss, lower, upper, p0, alg; autodiff=autodiff)
	return result
end

function fit_data(wave, xdata, fn, p0, alg, autodiff)
	sumsq_loss = get_sumsq_loss_fn(fn, wave, xdata)
	result = optimize(sumsq_loss, p0, alg; autodiff = autodiff)
	return result
end

function plot_comparison(;file_name="../WilsonCowanModel/data/wave_example.jld2", wave_fn=nothing,
		p0=nothing, lower=zero(p0), upper=(zero(p0).+Inf), alg=Fminbox(BFGS()), autodiff=:finite)
	wave, x, t = get_wave_data(file_name)
	xdata = construct_xdata(x,t)
	result= fit_data(wave, xdata, wave_fn, p0, lower, upper, alg, autodiff)
	data_plot = plot(wave, legend=false)
	fit_plot = plot(wave_fn(xdata, result.minimizer)', legend=false)
	combined_plot = plot(data_plot, fit_plot, layout=(2,1))
	savefig("$(Dates.now())")
	return result
end

function plot_comparison_particle_swarm(;file_name="../WilsonCowanModel/data/wave_example.jld2", wave_fn=nothing,
		p0=nothing, lower=zero(p0), upper=(zero(p0).+Inf))
	wave, x, t = get_wave_data(file_name)
	xdata = construct_xdata(x,t)
	@show p0
	@show lower
	result= fit_data(wave, xdata, wave_fn, p0, ParticleSwarm(lower=lower, upper=upper))
	data_plot = plot(wave, legend=false)
	fit_plot = plot(wave_fn(xdata, result.minimizer)', legend=false)
	combined_plot = plot(data_plot, fit_plot, layout=(2,1))
	savefig("$(Dates.now())")
	return result
end


end # module
