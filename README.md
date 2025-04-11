
# Analyzing Society's Spending Habits Using Economic Indicators

In this study, we aim to analyze the spending habits of society. We utilized data from the European Central Bank, Eurostat, the World Bank, and the OECD. Retail volume was chosen as the key indicator of society’s buying habits, as it increases when people make purchases. The data used was primarily on a monthly basis.

To explain retail volume, we considered the following variables:
- **Yearly population data**, interpolated linearly to monthly data
- **Monthly interest rates**
- **Monthly Consumer Price Index (CPI)**
- **Monthly unemployment rates**
- **Monthly Consumer Barometer parameter**

## Methodology

We created a comprehensive database by integrating all relevant datasets and identified the country with the most complete data: France, which was chosen for this study.

To determine which variables are most useful for analyzing retail volume, we first conducted a linear regression analysis. We then evaluated the results using ANOVA (Analysis of Variance) to assess the significance of each parameter.

