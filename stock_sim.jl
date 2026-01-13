using Random, Statistics



 #Geometric brownian motion model 

function stock_trial(rng, curr_price::Float64, vol::Float64; dt::Float64=1.0, d::Float64 = 0.0)
    Z = randn(rng)  # standard normal
    return curr_price * exp((d -0.5 * vol^2) * dt + vol * sqrt(dt) * Z)
end



#monte carlo simulator
function monte_carlo( curr_price::Float64, vol::Float64; T::Float64, dt::Float64, d::Float64=0.0, N::Int, seed::Union{Nothing,Int}=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed) #tracking the random sequence
    num_step = Int(round(T / dt))
    P = Vector{Float64}(undef, N) #undef makes sure array values are already undefined, in order to save time 
    @inbounds for i in 1:N # inbounds makes sure that julia doesn't check if i is in the array bounds for evry iteration to save time
        S = curr_price
       @inbounds for _ in 1:num_step  # second loop for days, uses the GBM model for recursion of price within num_step
            S = stock_trial(rng, S, vol; dt=dt, d=d)
        end
        P[i] = S
    end

    μ = mean(P) #mean
    σ = std(P) #standard deviation
    se = σ / sqrt(N) #standard error



    return μ, se, P
   


end

#curr_price = current stock price
#vol=volatility of stock
#d = drift (not a parameter currently (set to 0), but here if we decide to use it)
#dt = 1/252 # time per step 
#T = number of years entered by user
#num_step = T/dt# int number of steps like 1 year * 252 days
#N = #number of simulations in monte carlo model
#P = array of stock prices
#Z = standard normal to account for unpredictable movement in stocks per step (used a random small value)


