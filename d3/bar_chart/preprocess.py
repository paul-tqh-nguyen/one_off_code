#!/usr/bin/python3

import pandas as pd

df = pd.read_csv("./WPP2019_TotalPopulationBySex.csv")

# @hack grabs single word locations only in an attempt to get the country

shortened_df = df[df.Time==2019][df.Location.str.match(r'^[A-Z][a-z]*$')][['Location','PopTotal']].sort_values('PopTotal', ascending=False).head(10).to_json("location_populations.json", orient='records')

shortened_df.to_csv("./location_populations.csv", index=False)
