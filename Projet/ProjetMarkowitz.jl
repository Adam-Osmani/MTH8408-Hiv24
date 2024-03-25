using YahooFinance, ADNLPModels, LinearAlgebra


# Dates de début et de fin pour la récupération des données
start_date = "2018-12-13"
end_date = "2019-06-28"
N = 100 # Number of datapoints

# Récupération des données pour GOOG et BTC-CAD
data_goog = get_symbols("GOOG", start_date, end_date)
data_btc_cad = get_symbols("BTC-CAD", start_date, end_date)

# Truncate outputs
data_goog = values(data_goog["Close"][1:N])
data_btc_cad = values(data_btc_cad["Close"][1:N])

println(data_goog)
println(data_btc_cad)

# Calcul des rendements à chaque groupe de donéés
returns_goog = diff(data_goog) ./ data_goog[1:end-1]
returns_btc_cad = diff(data_btc_cad) ./ data_btc_cad[1:end-1]

#println(returns_good)
#println(returns_btc_cad)

returns_matrix = hcat(returns_goog, returns_btc_cad)
function riskreturn(x, R)
    r = mean(R, dims=1) # Calculer la moyenne 
    C = cov(R) # Calculer la covariance 
    risk = x' * C * x # Calcul du risque
    reward = -dot(r, x) # Calcul du rendement
    output = reward + μ*risk # μ doit être défini
    
    return output
end
