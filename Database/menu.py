import pandas as pd
import requests
import re
import psycopg2
import random

def parse_item(text):
    text=text.strip()
    match=re.match(r"^(\d+)\s*(.*)$",text)
    if match:
        quantity=int(match.group(1))
        name =match.group(2).strip()
    else :
        quantity=1
        name=text

    return name, quantity

url="https://api.kfcvietnam.com.vn/menu/kfcvn-generic-menu"
response=requests.get(url)
data=response.json()

db_config={
    'dbname' : 'recsys_db',
    'user' : 'postgres',
    'password' : '310506',
}
conn=psycopg2.connect(**db_config)
cursor=conn.cursor()

sub_cat=data['categories'][0]['categories']

for cat in sub_cat:
    for d in cat.get('dname',[]):
        if d.get('lang')=='vn':
            name=d['value']
            cursor.execute(
                """
                INSERT INTO categories (name)
                VALUES(%s)
                """,
                (name.strip(),)
            )
            cat_id = cursor.fetchone()[0]
            break
    products=cat['products'] or []
    for product in products:
        items=product['items'] or []
        for item in items:
            name=''
            is_combo=False
            is_bestsell=False
            is_seasonal=False
            is_hotsell=False
            for d in item.get('dname',[]):
                if d.get('lang')=='vn':
                    name=d['value']
            price=item['price']
            strikePrice=item['strikePrice'] or 0
            type=item['categoryName']
            if type in ('Hot Deal', "Value Deals", 'New Product') or ( strikePrice >0  and strikePrice  > price):
                is_seasonal=True
            elif type in ('For One','Sharing'):
                is_combo=True
            desciption=product['description'] or []
            num_meal=[]
            for d in desciption:
                if d.get('lang')== 'vn':
                    meal=d.get('value')
                    group=meal.split("+")
                    for g in group:
                        option=g.split("/")
                        for op in option:
                            item_name, quantity=parse_item(op)
                            num_meal.append({item_name : quantity})
            if len(num_meal) > 1:
                is_combo=True
            elif len(num_meal)==1 and next(iter(num_meal[0].values())) > 1:
                is_combo=True
            cursor.execute(
                """
                INSERT INTO meals (cat_id, name, price, available, is_hotsell, is_bestsell, is_seasonal, is_combo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (cat_id, name, price, True, is_hotsell, is_bestsell, is_seasonal, is_combo)
            )
            meal_id = cursor.fetchone()[0]
            modgrps=item.get('modgrps') or []
            for grp in modgrps:
                grp_name=grp['name']
                for d in grp.get('dname',[]):
                    if d.get('lang')=='vn':
                        grp_name=d['value'].strip()
                min_choices=grp.get('min',1)
                max_choices=grp.get('max',1)

                cursor.execute(
                    """
                    INSERT INTO option_groups (meal_id, name, min_choices, max_choices)
                    VALUES (%s, %s, %s, %s) RETURNING id
                    """,
                    (meal_id, grp_name, min_choices, max_choices)
                )
                group_id=cursor.fetchone()[0]

                for mod in grp.get('modifiers') or []:
                    mod_name=mod['name']
                    for d in mod.get('dname',[]):
                        if d.get('lang')=='vn':
                            mod_name=d['value']
                    item_name, quantity=parse_item(mod_name) 
                    cursor.execute("SELECT id FROM items WHERE name=%s", (item_name,))
                    row=cursor.fetchone()
                    if row:
                        item_id=row[0]
                    else:
                        cursor.execute("INSERT INTO items (name) VALUES (%s) RETURNING id", (item_name,))
                        item_id=cursor.fetchone()[0]

                    cursor.execute(
                        """
                        INSERT INTO meals_item (meal_id, item_id, quantity, group_id)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (meal_id, item_id, group_id) DO NOTHING
                        """,
                        (meal_id, item_id, quantity, group_id)
                    )
                    for sub_grp in mod.get('modgrps') or []:
                        sub_name=sub_grp['name']
                        for d in sub_grp.get('dname',[]):
                            if d.get('lang')=='vn':
                                sub_name=d['value'].strip()
                        sub_name=item_name + ' - ' + sub_name     
                        cursor.execute(
                            """
                            INSERT INTO option_groups (meal_id, name, min_choices, max_choices)
                            VALUES (%s, %s, %s, %s) RETURNING id
                            """,
                            (meal_id, sub_name, sub_grp.get('min',1), sub_grp.get('max',1))
                        )
                        sub_group_id=cursor.fetchone()[0]

                        for sub_mod in sub_grp.get('modifiers') or []:
                            sub_item=sub_mod['name']
                            for d in sub_mod.get('dname',[]):
                                if d.get('lang')=='vn':
                                    sub_item=d['value'].strip()   # "Gà Giòn Cay"

                            cursor.execute("SELECT id FROM items WHERE name=%s", (sub_item,))
                            row=cursor.fetchone()
                            if row:
                                sub_item_id=row[0]
                            else:
                                cursor.execute("INSERT INTO items (name) VALUES (%s) RETURNING id", (sub_item,))
                                sub_item_id=cursor.fetchone()[0]

                            cursor.execute(
                                """
                                INSERT INTO meals_item (meal_id, item_id, quantity, group_id)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (meal_id, item_id, group_id) DO NOTHING
                                """,
                                (meal_id, sub_item_id, 1, sub_group_id)
                            )
conn.commit()
cursor.close()
conn.close()


