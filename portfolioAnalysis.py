import numpy as np
import yfinance as yf
import csv
import datetime as dt
import os

class Stock:
    def __init__(self, stockName):
        self.stockName = stockName.upper()
        self.ticker = yf.Ticker(stockName.upper())
    
    def getTotalDividendsSince(self, date):
        dividends = self.ticker.dividends
        recent = np.sum(np.array(dividends[dividends.index >= date]))
        return recent
    
    def getCurrentPrice(self):
        self.currentPrice = self.ticker.info['currentPrice']
        return self.currentPrice
    
    def buyPrice(self, increaseBy = 0.03):
        return self.getCurrentPrice()+increaseBy
    
    def sellPrice(self, decreaseBy = 0.03):
        return self.getCurrentPrice()-decreaseBy
    
    def buyOn(self, dateYear = "2025", dateMonth = "01", dateDay = "01", increaseBy = 0.03):
        if (int(dateMonth)<10 and dateMonth[0] != "0") : dateMonth = "0"+dateMonth
        if (int(dateDay)<10 and dateDay[0] != "0") : dateDay = "0"+dateDay
        
        date = dateYear + "-" + dateMonth + "-" + dateDay
        endDate = dateYear + "-" + dateMonth + "-" + str(int(dateDay)+1)
        self.buyDate = date
        if (increaseBy <= .2 and self.getCurrentPrice() > .6):
            buyPrice = np.mean(np.array(self.ticker.history(start=date, end=endDate)['Close']))+ increaseBy
            return buyPrice
        elif (increaseBy >.2) : 
            return increaseBy
        else:
            return np.mean(np.array(self.ticker.history(start=date, end=endDate, interval="15m"))['Close'])+increaseBy

def initializeCSV(csvFileName = "index.csv"):
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(csvFileName):
        with open(csvFileName, "w", newline='') as csvFile:
            csv_writer = csv.writer(csvFile)
            csv_writer.writerow(["Stock", "Quantity", "Buy_Price", "Buy_Date", "Sell_Date", "Sell_Price"])

def totalReturnStock(stock: Stock, csvFileName = "index.csv"):
    with open(csvFileName, "r") as csvFile :
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        for line in csv_reader:
            if (line[0] == stock.stockName) :
                boughtAt = float(line[2])
                boughtDate= line[3]
        totalReturns =  (stock.getCurrentPrice() - boughtAt + stock.getTotalDividendsSince(boughtDate))/boughtAt
        return totalReturns*100

def totalReturnFull(csvFileName = "index.csv"):
    with open(csvFileName, "r") as csvFile :
        csv_reader = csv.reader(csvFile)
        boughtAt = 0.0
        currentPrice = 0.0
        totalDividends = 0.0
        next(csv_reader)
        for line in csv_reader:
                boughtAt += float(line[2]) * int(line[1])  # Include quantity
                boughtDate= line[3]
                stock = Stock(line[0])
                currentPrice += stock.getCurrentPrice() * int(line[1])  # Include quantity
                totalDividends += stock.getTotalDividendsSince(boughtDate) * int(line[1])  # Include quantity
        totalReturns =  (currentPrice - boughtAt + totalDividends)/boughtAt
        return totalReturns*100

def CAGR(csvFileName = "./index.csv"):
    with open(csvFileName, "r") as csvFile:
        csv_reader = csv.reader(csvFile)
        cagr = []
        next(csv_reader)
        for line in csv_reader:
            boughtAt = float(line[2])
            stock = Stock(line[0])
            
            boughtDate = dt.datetime.strptime(line[3], "%Y-%m-%d")
            if line[4]:
                sellDate = dt.datetime.strptime(line[4], "%Y-%m-%d")
            else:
                sellDate = dt.datetime.today()
            if line[5]:
                currentPrice = float(line[5])
            else:
                currentPrice = stock.getCurrentPrice()
            delta = sellDate - boughtDate
            timeInYears = delta.days / 365
            if timeInYears <= 0: continue
            cagrCurrentStock = (currentPrice / boughtAt)**(1/timeInYears)
            cagrCurrentStock -= 1
            cagr.append(cagrCurrentStock)
        return np.average(cagr) * 100

def weightedPortfolioReturn(csvFileName = "index.csv"):
    """Calculate weighted portfolio return based on investment amounts"""
    with open(csvFileName, "r") as csvFile:
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        
        totalInvestment = 0.0
        weightedReturns = 0.0
        
        for line in csv_reader:
            if not line: continue
            
            stock = Stock(line[0])
            quantity = int(line[1])
            buyPrice = float(line[2])
            buyDate = line[3]
            
            investment = quantity * buyPrice
            totalInvestment += investment
            
            # Calculate return for this stock
            currentValue = quantity * stock.getCurrentPrice()
            dividends = quantity * stock.getTotalDividendsSince(buyDate)
            stockReturn = ((currentValue + dividends - investment) / investment) * 100
            
            # Weight the return by investment amount
            weightedReturns += stockReturn * investment
        
        if totalInvestment > 0:
            return weightedReturns / totalInvestment
        return 0.0

def portfolioVolatility(csvFileName = "index.csv", days = 252):
    """Calculate portfolio volatility (standard deviation of returns)"""
    stocks = []
    weights = []
    
    with open(csvFileName, "r") as csvFile:
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        
        totalValue = 0.0
        stockData = []
        
        for line in csv_reader:
            if not line: continue
            
            stock = Stock(line[0])
            quantity = int(line[1])
            currentValue = quantity * stock.getCurrentPrice()
            totalValue += currentValue
            stockData.append((stock.stockName, currentValue))
        
        # Calculate weights
        for stockName, value in stockData:
            weights.append(value / totalValue)
            stocks.append(stockName)
    
    if not stocks:
        return 0.0
    
    # Get historical data for all stocks
    returns = []
    for stock in stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1y")
        if len(hist) > 1:
            stockReturns = hist['Close'].pct_change().dropna()
            returns.append(stockReturns.values)
    
    if not returns:
        return 0.0
    
    # Calculate portfolio returns
    minLength = min(len(r) for r in returns)
    portfolioReturns = np.zeros(minLength)
    
    for i, weight in enumerate(weights[:len(returns)]):
        portfolioReturns += weight * returns[i][:minLength]
    
    # Calculate annualized volatility
    volatility = np.std(portfolioReturns) * np.sqrt(days)
    return volatility * 100

def valueAtRisk(csvFileName = "index.csv", confidenceLevel = 0.05, days = 252):
    """Calculate Value at Risk (VaR) at given confidence level"""
    stocks = []
    weights = []
    
    with open(csvFileName, "r") as csvFile:
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        
        totalValue = 0.0
        stockData = []
        
        for line in csv_reader:
            if not line: continue
            
            stock = Stock(line[0])
            quantity = int(line[1])
            currentValue = quantity * stock.getCurrentPrice()
            totalValue += currentValue
            stockData.append((stock.stockName, currentValue))
        
        # Calculate weights
        for stockName, value in stockData:
            weights.append(value / totalValue)
            stocks.append(stockName)
    
    if not stocks:
        return 0.0, 0.0
    
    # Get historical data for all stocks
    returns = []
    for stock in stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1y")
        if len(hist) > 1:
            stockReturns = hist['Close'].pct_change().dropna()
            returns.append(stockReturns.values)
    
    if not returns:
        return 0.0, 0.0
    
    # Calculate portfolio returns
    minLength = min(len(r) for r in returns)
    portfolioReturns = np.zeros(minLength)
    
    for i, weight in enumerate(weights[:len(returns)]):
        portfolioReturns += weight * returns[i][:minLength]
    
    # Calculate VaR
    var = np.percentile(portfolioReturns, confidenceLevel * 100)
    varDollar = var * totalValue
    
    return var * 100, varDollar

def showProfitLoss(csvFileName = "index.csv"):
    """Show detailed profit/loss for each stock and overall portfolio"""
    print("\n" + "="*80)
    print("PROFIT & LOSS STATEMENT")
    print("="*80)
    
    totalInvested = 0.0
    totalCurrentValue = 0.0
    totalDividends = 0.0
    
    with open(csvFileName, "r") as csvFile:
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        
        print(f"{'Stock':<8} {'Qty':<6} {'Buy Price':<10} {'Current':<10} {'Invested':<12} {'Current Val':<12} {'Dividends':<10} {'P/L':<10} {'P/L %':<8}")
        print("-"*80)
        
        for line in csv_reader:
            if not line: continue
            
            stock = Stock(line[0])
            quantity = int(line[1])
            buyPrice = float(line[2])
            buyDate = line[3]
            
            currentPrice = stock.getCurrentPrice()
            invested = quantity * buyPrice
            currentValue = quantity * currentPrice
            dividends = quantity * stock.getTotalDividendsSince(buyDate)
            
            profitLoss = currentValue + dividends - invested
            profitLossPercent = (profitLoss / invested) * 100 if invested > 0 else 0
            
            totalInvested += invested
            totalCurrentValue += currentValue
            totalDividends += dividends
            
            print(f"{stock.stockName:<8} {quantity:<6} ${buyPrice:<9.2f} ${currentPrice:<9.2f} ${invested:<11.2f} ${currentValue:<11.2f} ${dividends:<9.2f} ${profitLoss:<9.2f} {profitLossPercent:<7.1f}%")
    
    totalPL = totalCurrentValue + totalDividends - totalInvested
    totalPLPercent = (totalPL / totalInvested) * 100 if totalInvested > 0 else 0
    
    print("-"*80)
    print(f"{'TOTAL':<8} {'':<6} {'':<10} {'':<10} ${totalInvested:<11.2f} ${totalCurrentValue:<11.2f} ${totalDividends:<9.2f} ${totalPL:<9.2f} {totalPLPercent:<7.1f}%")
    print("="*80)

def showEverything(csvFileName = "index.csv"):
    """Display comprehensive portfolio analysis"""
    print("\n" + "="*60)
    print("COMPREHENSIVE PORTFOLIO ANALYSIS")
    print("="*60)
    
    try:
        # Show P/L first
        showProfitLoss(csvFileName)
        
        # Portfolio metrics
        print("\nPORTFOLIO METRICS:")
        print("-"*30)
        print(f"Total Portfolio Return: {totalReturnFull(csvFileName):.2f}%")
        print(f"Portfolio CAGR: {CAGR(csvFileName):.2f}%")
        print(f"Weighted Portfolio Return: {weightedPortfolioReturn(csvFileName):.2f}%")
        print(f"Portfolio Volatility: {portfolioVolatility(csvFileName):.2f}%")
        
        var_percent, var_dollar = valueAtRisk(csvFileName)
        print(f"Value at Risk (5%): {var_percent:.2f}% (${var_dollar:.2f})")
        
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")

def displayMenu():
    print("\n" + "="*50)
    print("STOCK PORTFOLIO MANAGEMENT SYSTEM")
    print("="*50)
    print("1.  Buy stock (current date)")
    print("2.  Buy stock (specific date)")
    print("3.  Sell stock")
    print("4.  Check total returns of a stock")
    print("5.  Check total returns of portfolio")
    print("6.  Check CAGR of full portfolio")
    print("7.  Check Weighted Portfolio Return")
    print("8.  Check Standard Deviation (Volatility)")
    print("9.  Check Value at Risk (VaR)")
    print("10. Show P/L overall")
    print("11. Show everything")
    print("12. Exit")
    print("="*50)

# Initialize CSV file
initializeCSV()

choice = 0

while (choice != 12):
    displayMenu()
    try:
        choice = int(input("What do you want to do/see [1-12]: ")) or 13
    except ValueError:
        choice = 13
    
    match choice:
        case 1:
            stockSign = input("Enter the symbol of stock you want to add (AAPL): ") or "AAPL"
            stock = Stock(stockSign)
            stockQty = int(input("How many stocks you have bought: "))
            buyAsk = input("Enter the buy Price\nor enter the increaseBy (0.03): ") or "0.03"
            ask = float(buyAsk)
            today = dt.date.today().strftime("%Y-%m-%d")
            with open("./index.csv", "a", newline='') as csvFile:
                csv_writer = csv.writer(csvFile)
                csv_writer.writerow([stock.stockName, stockQty, stock.buyPrice(ask), today, "", 0])
            print(f"Added {stockQty} shares of {stock.stockName} at ${stock.buyPrice(ask):.2f}")
        
        case 2: 
            stockSign = input("Enter the symbol of stock you want to add (AAPL): ") or "AAPL"
            stock = Stock(stockSign)
            stockQty = int(input("How many stocks you have bought: "))
            buyAsk = input("Enter the buy Price\nor enter the increaseBy (0.03): ") or "0.03"
            ask = float(buyAsk)
            dateYear = input("Enter the year you bought it in YYYY format: ")
            dateMonth = input("Enter the month you bought it in MM format: ")
            dateDay = input("Enter the day you bought it in DD format: ")
            date = dateYear + "-" + dateMonth + "-" + dateDay
            with open("./index.csv", "a", newline='') as csvFile:
                csv_writer = csv.writer(csvFile)
                csv_writer.writerow([stock.stockName, stockQty, stock.buyOn(dateYear, dateMonth, dateDay, ask), date, "", 0])
            print(f"Added {stockQty} shares of {stock.stockName} bought on {date}")
        
        case 3:
            stockSign = input("Enter the symbol of stock you want to sell (AAPL): ") or "AAPL"
            found = False
            stock = Stock(stockSign)
            allRows = []
            with open("./index.csv", "r") as csvFile:
                csvReader = csv.reader(csvFile)
                for row in csvReader:
                    if (row[0] == stock.stockName):
                        temp = row
                        temp[4] = dt.date.today().strftime("%Y-%m-%d")
                        temp[5] = str(stock.getCurrentPrice())
                        found = True 
                        allRows.append(temp)
                    else:
                        allRows.append(row)
            
            if (found):
                print(f"Selling {stock.stockName} at current price ${stock.getCurrentPrice():.2f}")
                with open("./index.csv", "w", newline="") as csvFile:
                    csvWriter = csv.writer(csvFile)
                    csvWriter.writerows(allRows)
            else:
                print(f"Stock {stock.stockName} not found in portfolio")

        case 4:
            stockSign = input("Enter the symbol of stock you want to check: ") or "AAPL"
            stock = Stock(stockSign)
            try:
                with open("./index.csv", "r") as csvFile:
                    csvReader = csv.reader(csvFile)
                    next(csvReader)  # Skip header
                    found = False
                    for row in csvReader:
                        if row[0] == stock.stockName:
                            boughtAt = float(row[2])
                            found = True
                            break
                    
                    if found:
                        print(f"Current price of {stock.stockName}: ${stock.getCurrentPrice():.2f}")
                        print(f"Bought at price: ${boughtAt:.2f}")
                        print(f"Total returns on {stock.stockName}: {totalReturnStock(stock):.2f}%")
                    else:
                        print(f"Stock {stock.stockName} not found in portfolio")
            except Exception as e:
                print(f"Error: {e}")
        
        case 5:
            try:
                print(f"Total Portfolio Return: {totalReturnFull():.2f}%")
            except Exception as e:
                print(f"Error calculating portfolio return: {e}")
        
        case 6:
            try:
                print(f"Portfolio CAGR: {CAGR():.2f}%")
            except Exception as e:
                print(f"Error calculating CAGR: {e}")
        
        case 7:
            try:
                print(f"Weighted Portfolio Return: {weightedPortfolioReturn():.2f}%")
            except Exception as e:
                print(f"Error calculating weighted return: {e}")
        
        case 8:
            try:
                print(f"Portfolio Volatility (Standard Deviation): {portfolioVolatility():.2f}%")
            except Exception as e:
                print(f"Error calculating volatility: {e}")
        
        case 9:
            try:
                var_percent, var_dollar = valueAtRisk()
                print(f"Value at Risk (5% confidence): {var_percent:.2f}% (${var_dollar:.2f})")
            except Exception as e:
                print(f"Error calculating VaR: {e}")
        
        case 10:
            try:
                showProfitLoss()
            except Exception as e:
                print(f"Error showing P/L: {e}")
        
        case 11:
            try:
                showEverything()
            except Exception as e:
                print(f"Error showing complete analysis: {e}")
        
        case 12:
            print("Goodbye!")
            break
        
        case _:
            print("Invalid choice. Please enter a number between 1-12.")

print("Thank you for using the Stock Portfolio Management System!")