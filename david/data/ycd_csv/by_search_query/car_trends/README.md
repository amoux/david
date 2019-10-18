

# Vehicle car trends dataset


# PRODIGY COMMANDS EXECUTED FOR THIS CORPUS

 **creating the prodigy dataset for cartrends**

(prodigy) ego@neural-machine:~/vuepoint_dev/prodigy$ prodigy dataset cartrends "A dataset containing over 17k comments scraped from youtube, about car trends" --author Carlos_Segura

  ✨  Successfully added 'cartrends' to database SQLite.

 **textcat.teach label PRODUCTS if comment talks about products**

(prodigy) ego@neural-machine:~/vuepoint_dev/prodigy$ prodigy textcat.teach cartrends en_core_web_lg working_datasets/car_trends/cartrends.json --label PRODUCTS
Using 1 labels: PRODUCTS

  ✨  Starting the web server at http://localhost:8080 ...
  Open the app in your browser and start annotating!

(prodigy) ego@neural-machine:~/vuepoint_dev/prodigy$ prodigy textcat.batch-train cartrends --output cartrends-model --eval-split 0.2



