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

for cat in sub_cat:
    products=cat['products'] or []
    for product in products:
        items=product['items'] or []
        for item in items:
            name=''
            is_combo=False
            is_bestsell=False
            is_seasonal=False
            is_hotsell=Fale
            for d in item.get('dname',[]):
                if d.get('lang')=='vn':
                    name=d['value']
            price=item['price']
            strikePrice=item['strikePrice']
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


conn.commit()
cursor.close()
conn.close()


