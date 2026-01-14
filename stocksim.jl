using Statistics
using Base.Threads
using Plots
using Printf 
using Random

abstract type EuropeanOption end

# define call option - when the right time to BUY is @ specific price
struct CallOption <: EuropeanOption
    strike::Float64
end

# define put option - when the right time to SELL is @ specific price
struct PutOption <: EuropeanOption
    strike::Float64
end

payoff(opt::CallOption, spot_price::Float64) = max(spot_price - opt.strike, 0.0)
payoff(opt::PutOption, spot_price::Float64)  = max(opt.strike - spot_price, 0.0)

# predicts next price based on drift and volatility
function stock_trial(rng, curr_price::Float64, vol::Float64; dt::Float64=1.0, d::Float64 = 0.0)
    Z = randn(rng)
    return curr_price * exp((d - 0.5 * vol^2) * dt + vol * sqrt(dt) * Z)
end

function run_monte_carlo(option::EuropeanOption, S0, r, σ, T, dt, N)
    payoffs = Vector{Float64}(undef, N)
    num_steps = Int(round(T / dt))

    # @threads distributes X amount of simulations across CPU cores
    @threads for i in 1:N
        rng = Random.default_rng() 
        S = S0
        
        for _ in 1:num_steps
            S = stock_trial(rng, S, σ; dt=dt, d=r)
        end
        
        payoffs[i] = payoff(option, S)
    end
    
    price = mean(payoffs) * exp(-r * T)
    return price, payoffs
end

function main()
    println("Option Pricing Input")

    print("Initial stock price: ")
    S0 = parse(Float64, readline()) # curr price

    print("Strike price (K): ")
    K = parse(Float64, readline()) # "target" price in the option contract

    print("Risk-free rate (r) [e.g. 0.05]: ")
    r = parse(Float64, readline()) # % we assume the stock should grow annually

    print("Volatility (σ) [e.g. 0.2]: ")
    σ = parse(Float64, readline()) # how much the stock swings

    print("Years to expiry (T): ")
    T = parse(Float64, readline()) # length of simulation

    print("Number of simulations (N): ")
    N = parse(Int, readline()) # how many different stock price paths to simulate
    
    dt = 1/252 # roughly 252 trading days/year
    
    println("\nUsing $(nthreads()) CPU threads...")
    
    call_opt = CallOption(K) # when to BUY (stock goes up)
    println("Simulating Call Option...")
    @time call_price, call_data = run_monte_carlo(call_opt, S0, r, σ, T, dt, N)
    
    put_opt = PutOption(K) # when to SELL (stock goes down)
    println("Simulating Put Option...")
    @time put_price, put_data = run_monte_carlo(put_opt, S0, r, σ, T, dt, N)

    println("\n" * "="^30)
    @printf("Call Price: \$%.4f\n", call_price) # fair price based on avg of all payoffs
    @printf("Put Price:  \$%.4f\n", put_price)
    println("="^30)

    println("\nGenerating Graph...")
    running_avg = cumsum(call_data) ./ (1:N) .* exp(-r * T)
    
    p = plot(running_avg[1:min(N, 20000)], 
             title="Monte Carlo Convergence (Call Option)",
             xlabel="Simulations", ylabel="Price Estimate",
             lw=2, label="Running Average")
    hline!([call_price], label="Final Price", linestyle=:dash, color=:red)
    
    savefig(p, "convergence_plot.png")
    println("Graph is saved as 'convergence_plot.png'")
end

main()