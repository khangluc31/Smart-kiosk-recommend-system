import psycopg2
import numpy as np
import hashlib
from unregular import linUCB
name=''
phone=''
db_config={
    'dbname' : 'recsys_db',
    'user' : 'postgres',
    'password' : '310506',
}
conn=psycopg2.connect(**db_config)
def create_user_id(name, phone):
    text = f"{name.strip()}:{phone.strip()}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

user_id = create_user_id(name,phone)
try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT num_buy
            FROM users
            WHERE id= %s
        """, (user_id,))
        num_buy=cur.fetchone()

    if(num_buy[0] <11 or num_buy==None):
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id is_hotsell is_bestsell available num_sold
                FROM meals
            """)
            meals = cur.fetchall()
        meal_ids = [meal[0] for meal in meals]
        features=[]
        for meal in meals:
            hotsell, bestsell, available, sold=meal
            x=np.array(
                1.0,
                float(hotsell),
                float(bestsell),
                float(available),
                float(sold)
            )
            features.append(x)
        model=linUCB(len(meals),5,1)
        suggest_meal=model.select_arm(features)
        candidate = [meals[i] for i in suggest_meal]


finally:
    conn.close()



