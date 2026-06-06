# %% [markdown]
# # 🏠 House Price Prediction — EDA
# **Dataset:** Housing.csv · 545 records · 13 features · Prices in ₹ (INR)

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("../data/housing.csv")

# Encode binary columns for correlation analysis
binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in binary_cols:
    df[col + '_enc'] = (df[col].str.lower() == 'yes').astype(int)

df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0,1)
df['total_rooms']      = df['bedrooms'] + df['bathrooms']

print(df.shape)
df.head()

# %% [markdown]
# ## 1. Dataset Overview

# %%
df.info()
df.describe()

# %% [markdown]
# ## 2. Price Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(df['price'], bins=30, kde=True, ax=axes[0])
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
axes[0].set_title("Price Distribution")

sns.histplot(np.log(df['price']), bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].set_title("Log(Price) Distribution")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()
# Price is right-skewed — log transform helps linear models

# %% [markdown]
# ## 3. Correlation Heatmap

# %%
num_cols = ['area','bedrooms','bathrooms','stories','parking',
            'mainroad_enc','guestroom_enc','basement_enc',
            'hotwaterheating_enc','airconditioning_enc','prefarea_enc',
            'area_per_bedroom','total_rooms','price']
corr = df[num_cols].corr()

plt.figure(figsize=(12,9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={'size':8})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# %% [markdown]
# ## 4. Price vs Area (Strongest Feature)

# %%
plt.figure(figsize=(8,5))
plt.scatter(df['area'], df['price']/1e6, alpha=0.4, s=20)
m, b = np.polyfit(df['area'], df['price']/1e6, 1)
xs = np.linspace(df['area'].min(), df['area'].max(), 100)
plt.plot(xs, m*xs+b, 'r--', lw=2, label=f'Trend line')
plt.xlabel("Area (sq ft)"); plt.ylabel("Price (₹ Millions)")
plt.title("Price vs Area")
plt.legend()
plt.tight_layout()
plt.savefig("price_vs_area.png")
plt.show()

# %% [markdown]
# ## 5. Price by Furnishing Status

# %%
plt.figure(figsize=(8,5))
order = df.groupby('furnishingstatus')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='furnishingstatus', y='price', order=order, palette='Set2')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
plt.title("Price by Furnishing Status")
plt.tight_layout()
plt.savefig("price_by_furnishing.png")
plt.show()

# %% [markdown]
# ## 6. Amenity Impact on Price

# %%
amenities = {
    'mainroad': 'Main Road', 'airconditioning': 'AC',
    'basement': 'Basement', 'guestroom': 'Guest Room',
    'hotwaterheating': 'Hot Water', 'prefarea': 'Preferred Area'
}
fig, axes = plt.subplots(2, 3, figsize=(14,8))
for ax, (col, label) in zip(axes.flatten(), amenities.items()):
    sns.boxplot(data=df, x=col, y='price', ax=ax, palette=['#ff9999','#66b2ff'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e6:.1f}M'))
    ax.set_title(label); ax.set_xlabel('')
plt.suptitle("Price by Amenity (yes vs no)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("amenity_impact.png")
plt.show()

# %% [markdown]
# ## 7. Key Insights
# - **Area** has the strongest positive correlation with price (~0.54)
# - **Furnished** homes command a premium over unfurnished
# - **AC, preferred area, hot water heating** each add significant value
# - Main road access increases price noticeably
# - Price is right-skewed — 75% of houses priced below ₹57L

print("✅ EDA complete")
