using Random, Statistics


curr_price = 0.0 #current stock price
vol=0.0 #volatility of stock
num_sim = 0 # int number of simulations 
#P = array of stock prices
#monte carlo simulator

function monte_carlo(trial, num_sim::Int; seed::Union{Nothing,Int}=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed) #tracking the random sequence

    P = Vector{Float64}(undef, N). #undef makes sure array values are already undefined, in order to avoid extra checks 

    @inbounds for i in 1:N # inbounds makes sure that julia doesn't check if i is in the array bounds for evry iteration to save time
        P[i] = trial(rng)
    end

    μ = mean(P) #mean
    σ = std(P) #standard deviation
    se = σ / sqrt(N) #standard error

    return μ, se, P
end