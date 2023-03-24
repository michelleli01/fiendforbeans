import csv

DATA_FILE_PATH = "./data/archive/simplified_coffee.csv"
INIT_SQL_PATH = "./init.sql"
DB_NAME = "coffeedb"
TABLE_NAME = "reviews"

TABLE_SCHEMA = {
    "id": "int",
    "name": "varchar(128)",
    "review": "varchar(1024)",
}


def write_init_sql(columns, data):
    column_idxs = []
    for idx, column_name in enumerate(TABLE_SCHEMA):
        if idx != 0:
            column_idxs.append(columns.index(column_name))

    file = open(INIT_SQL_PATH, "w")
    file.write(
        f"""
CREATE DATABASE IF NOT EXISTS {DB_NAME};

USE {DB_NAME};
DROP TABLE IF EXISTS {TABLE_NAME};

CREATE TABLE {TABLE_NAME}(
"""
    )

    for idx, (k, v) in enumerate(TABLE_SCHEMA.items()):
        schema = f",\n\t\t{k} {v}" if idx != 0 else f"{k} {v}"
        file.write(schema)
    file.write("\n);\n\n")

    for idx, row in enumerate(data):
        row_values = f"{idx + 1}"
        for idx in column_idxs:
            value = row[idx].replace("'", "''").replace("%", "%%")
            row_values += f",'{value}'"
        file.write(f"INSERT INTO {TABLE_NAME} VALUE({row_values});\n")
    file.close()


def main():
    file = open(DATA_FILE_PATH)
    csv_reader = csv.reader(file)
    columns = []
    data = []
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            columns = row
        else:
            data.append(row)
    file.close()
    write_init_sql(columns, data)


if __name__ == "__main__":
    main()
