import math  
import re
import joblib
import numpy as np
from flask import Flask, render_template, request
import scipy.stats as stats

app = Flask(__name__)

# Load trained model
model = joblib.load("distribution_model.pkl")

# Extract numerical values from the problem statement
def extract_parameters(problem_statement, distribution):
    numbers = list(map(float, re.findall(r'\d+\.?\d*', problem_statement)))

    if distribution == 'binomial' and len(numbers) >= 3:
        n, p, k = numbers[:3]
        return int(n), p / 100, int(k)  

    elif distribution == 'poisson' and len(numbers) >= 2:
        lam, k = numbers[:2]
        return float(lam), int(k)

    elif distribution == 'exponential' and len(numbers) >= 2:
        lam, x = numbers[:2]
        return float(lam), float(x)

    elif distribution == 'normal' and len(numbers) >= 3:
        mu, sigma, x = numbers[:3]
        return float(mu), float(sigma), float(x)

    elif distribution == 'uniform' and len(numbers) >= 3:
        a, b, x = numbers[:3]
        return float(a), float(b), float(x)

    elif distribution == 'geometric' and len(numbers) >= 2:
        p, k = numbers[:2]
        return float(p) / 100, int(k)  

    return None


# Solve the probability problem with step-by-step explanation
def solve_distribution_problem(distribution, parameters):
    if not parameters or None in parameters:
        return None, "❌ Insufficient parameters extracted."

    steps = []  

    if distribution == 'binomial':
        n, p, k = parameters
        comb = math.comb(n, k)  

        steps.append(f"Step 1: Identify Values")
        steps.append(f"n = {n}, p = {p}, k = {k}")

        steps.append(f"Step 2: Compute Binomial Coefficient")
        steps.append(f"C(n, k) = {comb}")

        steps.append(f"Step 3: Apply Binomial Probability Formula")
        steps.append(f"P(X = k) = C(n, k) * p^k * (1-p)^(n-k)")

        result = stats.binom.pmf(k, n, p)

        steps.append(f"Step 4: Compute Probability")
        steps.append(f"P(X = {k}) ≈ {result:.4f}")

    elif distribution == 'poisson':
        lam, k = parameters  

        poisson_factorial = math.factorial(k)  

        poisson_probability = ((lam ** k) * math.exp(-lam)) / poisson_factorial

        steps.append(f"Step 1: Given Values: λ = {lam}, k = {k}")
        steps.append(f"Step 2: Compute Factorial of k: {k}! = {poisson_factorial}")
        steps.append(f"Step 3: Apply Poisson Formula: P(X = k) = (λ^k * e^(-λ)) / k!")
        steps.append(f"Step 4: Compute Probability: P(X = {k}) ≈ {poisson_probability:.5f}")

        result = poisson_probability

    elif distribution == 'exponential':
        lam, x = parameters

        steps.append(f"Step 1: Identify Values")
        steps.append(f"λ = {lam}, x = {x}")

        steps.append(f"Step 2: Apply Exponential Probability Formula")
        steps.append(f"P(T ≤ t) = 1 - e^(-λt)")

        result = 1 - math.exp(-lam * x)  

        steps.append(f"Step 3: Compute Probability")
        steps.append(f"P(T ≤ {x}) ≈ {result:.4f}")

    elif distribution == 'normal':
        mu, sigma, x = parameters

        steps.append(f"Step 1: Identify Values")
        steps.append(f"μ = {mu}, σ = {sigma}, x = {x}")

        steps.append(f"Step 2: Compute Z-score")
        z = (x - mu) / sigma
        steps.append(f"Z = ({x} - {mu}) / {sigma} ≈ {z:.4f}")

        result = stats.norm.cdf(x, loc=mu, scale=sigma)

        steps.append(f"Step 3: Compute Probability")
        steps.append(f"P(X ≤ {x}) ≈ {result:.4f}")

    elif distribution == 'uniform':
        a, b, x = parameters

        steps.append(f"Step 1: Identify Values")
        steps.append(f"a = {a}, b = {b}, x = {x}")

        steps.append(f"Step 2: Apply Uniform Distribution Formula")
        steps.append(f"P(X ≤ x) = (x - a) / (b - a)")

        result = (x - a) / (b - a) if a <= x <= b else 0

        steps.append(f"Step 3: Compute Probability")
        steps.append(f"P(X ≤ {x}) ≈ {result:.4f}")

    elif distribution == 'geometric':
        p, k = parameters

        steps.append(f"Step 1: Identify Values")
        steps.append(f"p = {p}, k = {k}")

        steps.append(f"Step 2: Apply Geometric Probability Formula")
        steps.append(f"P(X = k) = (1-p)^(k-1) * p")

        result = (1 - p) ** (k - 1) * p

        steps.append(f"Step 3: Compute Probability")
        steps.append(f"P(X = {k}) ≈ {result:.4f}")

    else:
        return None, "❌ Unknown distribution."

    explanation = "<br>".join(steps)

    return result, explanation


@app.route("/", methods=["GET", "POST"])
def index():
    explanation = None
    distribution = None
    parameters = None
    result = None

    if request.method == "POST":
        problem = request.form.get("problem")  

        if problem:
            distribution = model.predict([problem])[0]
            parameters = extract_parameters(problem, distribution)

            if parameters and None not in parameters:
                result, explanation = solve_distribution_problem(distribution, parameters)
            else:
                explanation = "❌ Could not extract necessary parameters."

    return render_template("index.html", 
                           distribution=distribution.capitalize() if distribution else "Unknown",
                           parameters=parameters, 
                           result=result, 
                           explanation=explanation)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

#uyiopojgit branch
