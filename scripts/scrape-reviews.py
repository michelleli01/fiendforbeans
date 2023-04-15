import requests
import bs4
import shutil
import os
import csv

NUM_REVIEWS = 1500

WEBSITE_URL = "https://www.coffeereview.com/review/page/{}/?locations=na"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
}

GRAMS_TO_OUNCES = 0.035274

LABELS = {
    "Roast Level": "roast",
    "Est. Price": "dollars_per_ounce",
    "Review Date": "review_date",
    "Coffee Origin": "origin",
}

roasters = set()


def write_data_to_csv(data):
    print("Writing data to csv file...")
    with open("./data/archive/scraped-data.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        fields = list(data[0].keys())
        csv_writer.writerow(fields)
        for row in data:
            csv_row = [row[field] for field in fields]
            csv_writer.writerow(csv_row)


def get_unit_price(price_str):
    try:
        slash_idx = price_str.find("/")
        dollar_idx = price_str.find("$")
        if dollar_idx == -1:
            return None
        price = float(price_str[dollar_idx + 1 : slash_idx])
        num_ounces = -1
        if "ounce" in price_str:
            ounce_idx = price_str.find("ounce")
            num_ounces = float(price_str[slash_idx + 1 : ounce_idx - 1])
        else:
            grams_idx = price_str.find("gram")
            num_grams = float(price_str[slash_idx + 1 : grams_idx - 1])
            num_ounces = num_grams * GRAMS_TO_OUNCES
        return round(price / num_ounces, 2) if num_ounces != -1 else None
    except:
        return None


def save_roaster_image(image_url, file_name):
    res = requests.get(image_url, headers=HEADERS, stream=True)
    res.raw.decode_content = True
    with open(file_name, "wb") as file:
        shutil.copyfileobj(res.raw, file)
    print("Saved roaster logo!")


def get_data_from_review_page(link):
    res = requests.get(link, headers=HEADERS)
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    review_div = soup.find("div", class_="review-template")
    roaster_div = soup.find("div", class_="column col-3")
    roaster_link_elt = roaster_div.find("a")
    roaster = soup.find("p", class_="review-roaster").text
    roaster_link = None

    if roaster_link_elt is not None:
        roaster_link = roaster_link_elt["href"]
        img_tag = roaster_link_elt.find("img")
        if roaster not in roasters and img_tag is not None:
            img_link = img_tag["src"]
            save_roaster_image(
                img_link, f"./backend/static/images/roaster_logos/{roaster}.webp"
            )
    data = {
        "roaster": roaster,
        "name": soup.find("h1", class_="review-title").text,
        "review": review_div.h2.find_next_sibling("p").text.replace("\r\n", ""),
        "roaster_link": roaster_link,
    }

    for table_col in soup.find_all("table", class_="review-template-table"):
        rows = table_col.find_all("tr")
        for row in rows:
            label, value = row.find_all("td")
            if label.text[:-1] in LABELS:
                data[LABELS[label.text[:-1]]] = value.text
    data["dollars_per_ounce"] = get_unit_price(data["dollars_per_ounce"])
    return data


def main():
    if os.path.exists("./backend/static/images/roaster_logos"):
        shutil.rmtree("./backend/static/images/roaster_logos")
    os.mkdir("./backend/static/images/roaster_logos")

    data = []
    page = 1
    while len(data) < NUM_REVIEWS:
        url = WEBSITE_URL.format(page)
        print(url)
        res = requests.get(url, headers=HEADERS)
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        reviews = soup.find_all("div", class_="review-template")
        for review in reviews:
            review_link = review.find("h2", class_="review-title").find("a")["href"]
            data.append(get_data_from_review_page(review_link))
        page += 1
    write_data_to_csv(data)


if __name__ == "__main__":
    main()
