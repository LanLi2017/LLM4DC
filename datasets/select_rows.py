import pandas as pd

# Assuming you have a DataFrame named df
# For example:
# df = pd.read_csv('original/PPP_full_data.csv')
# df = pd.read_csv('original/Menu.csv')
df = pd.read_csv('original/Dish.csv')

# Shuffle the rows randomly
df_shuffled = df.sample(frac=1, random_state=42)

# Select the first n rows
# df_selected = df_shuffled.head(1000)
# df_selected = df_shuffled.head(100)
df_selected = df_shuffled.head(500)

# Optionally, you can save the selected rows to a new CSV file
# df_selected.to_csv('ppp_data.csv', index=False)
# df_selected.to_csv('menu_data.csv', index=False)
df_selected.to_csv('dish_data.csv', index=False)

print(df_selected)
