import csv
import re


def tokenize(text):
    """Returns a list of words that make up the text.
    Params: {text: String}
    Returns: List
    """
    return re.findall("[a-z]+", text.lower())


fruit = [
    "blackberry",
    "raspberry",
    "blueberry",
    "strawberry",
    "raisin",
    "prune",
    "coconut",
    "cherry",
    "pomegranate",
    "pineapple",
    "grape",
    "apple",
    "peach",
    "pear",
    "grapefruit",
    "orange",
    "lemon",
    "lime",
    "citrus",
    "berry",
    "fruit",
]

floral = ["jasmine", "rose", "chamomile", "tea", "floral"]

sweet = [
    "aromatic",
    "vanilla",
    "sugar",
    "honey",
    "caramelized",
    "maple",
    "syrup",
    "molasses",
    "sweet",
]

nutty = ["almond", "hazelnut", "peanuts", "nutty"]

cocoa = ["chocolate", "cocoa", "cacao"]

spice = ["clove", "cinnamon", "nutmeg", "anise", "pepper", "pungent", "spice"]

roasted = [
    "cereal",
    "malt",
    "grain",
    "brown",
    "roast",
    "burnt",
    "smoky",
    "ashy",
    "acrid",
    "tobacco",
]

chemical = ["chemical", "rubber", "medicinal", "salty", "bitter"]

papery = [
    "phenolic",
    "meaty",
    "brothy",
    "animalic",
    "musty",
    "earthy",
    "dusty",
    "damp",
    "woody",
    "papery",
    "cardboard",
    "stale",
]

flavor_cat = {
    "fruit": fruit,
    "floral": floral,
    "sweet": sweet,
    "nutty": nutty,
    "cocoa": cocoa,
    "spice": spice,
    "roasted": roasted,
    "chemical": chemical,
    "papery": papery,
}


rev_flavor_cat = {}
for key, value in flavor_cat.items():
    for v in range(len(value)):
        rev_flavor_cat[value[v]] = key


def main():
    data = []

    with open('./data/archive/scraped-data.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            review = row[2]
            tokens = tokenize(review)
            categories = set()
            for token in tokens:
                if token in rev_flavor_cat.keys():
                    categories.add(rev_flavor_cat[token])
            categories = list(categories)
            row.append(str(categories))
            data.append(row)

    with open('./data/archive/scraped-categories-2.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        columns = data[0][:-1]
        columns.append('flavor')
        csv_writer.writerow(columns)
        for row in data[1:]:
            csv_writer.writerow(row)


if __name__ == '__main__':
    main()
