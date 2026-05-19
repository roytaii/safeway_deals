# Safeway Weekly Ad Viewer

This website uses flipp's api to scrape each item on Safeway's Weekly Ad. 
It then takes each item, and appends it to `safeway_prices.csv`. Nothing should ever deleted from `safeway_prices.csv`, only added.
`safeway_visualization` reads `safeway_prices.csv` and converts it to a json file in `deals.json`. 
`deals.json` is then read to generate the website visuals. 
Since Safeway's Weekly Ad's change every week, I have github actions set up to run `scrape_safeway.py` every week. 
