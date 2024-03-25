using YahooFinance, ADNLPModels, LinearAlgebra, Statistics, Ipopt


# Dates de début et de fin pour la récupération des données
start_date = "2018-12-13"
end_date = "2019-06-28"
N = 100 # Number of datapoints

# Récupération des données pour GOOG et BTC-CAD
data_goog = get_symbols("GOOG", start_date, end_date)
data_btc_cad = get_symbols("BTC-CAD", start_date, end_date)
data_cad_usd = get_symbols("CADUSD=X", start_date, end_date)
data_cad_eur = get_symbols("CADEUR=X", start_date, end_date)

# Truncate outputs
data_goog = values(data_goog["Close"][1:N])
data_btc_cad = values(data_btc_cad["Close"][1:N])
data_cad_usd = values(data_cad_usd["Close"][1:N])
data_cad_eur = values(data_cad_eur["Close"][1:N])

# println(data_goog)
# println(data_btc_cad)

# Calcul des rendements à chaque groupe de donéés
# returns_goog = diff(data_goog) ./ data_goog[1:end-1]
# returns_btc_cad = diff(data_btc_cad) ./ data_btc_cad[1:end-1]

#println(returns_good)
#println(returns_btc_cad)

returns_matrix = hcat(data_goog, data_btc_cad,data_cad_usd,data_cad_eur)
function riskreturn(x, R, μ)
    p = 0.9
    l = Int(size(R,2))
    N = Int(size(R,1))
    r = zeros(l)
    for j in 1:l
        a = 0
        b = 0
        for i in 1:N
            a += p^(N-i) * log(R[i,j])
            b += p^(N-i)
        end
        r[j] = exp(a/b)
    end
    # r = mean(R, dims=1) # Calculer la moyenne
    C = cov(R) # Calculer la covariance
    risk = x' * C * x # Calcul du risque

    reward = -dot(r, x) # Calcul du rendement
    output = reward + μ*risk # μ doit être défini

    return output
end

l = Int(size(returns_matrix,2))

μ = 0.5
R = returns_matrix
f(x) = riskreturn(x, R, μ)
c(x) = sum(x)
x0 = ones(l)/l
lvar = zeros(l)
uvar = ones(l)
lcon = 1
ucon = 1
nlp = (f,x0,lvar,uvar,c,lcon,ucon)

stats = ipopt(nlp)