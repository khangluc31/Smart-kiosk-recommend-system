INSERT INTO users (num_buy, avg_spent)
SELECT 0,0
FROM generate_series(1,50);

INSERT INTO history(user_id, meal_id, times_order, day_buy)
SELECT
    u.id,
    m.id,
    floor(random()*20+1)::int,
    Now()-random() * INTERVAL '365 days' 
FROM users AS u
CROSS JOIN LATERAL(
    SELECT id
    FROM meals
    ORDER BY random()
    LIMIT 9
 ) AS m;

INSERT INTO user_fav_cat(user_id, cat_id)
SELECT
    u.id,
    c.id
FROM users as u
CROSS JOIN LATERAL(
    SELECT id
    FROM categories
    ORDER BY random()
    LIMIT 5
) AS c;

INSERT INTO user_fav_meal(user_id,meal_id)
SELECT
    u.id,
    m.id
FROM users as u
CROSS JOIN LATERAL(
    SELECT id
    FROM meals
    ORDER BY random()
    LIMIT 5
) AS m;
UPDATE users u
SET 
    num_buy = stats.total_buy,
    avg_spent = stats.avg_spent
FROM (
    SELECT 
        h.user_id,
        SUM(h.times_order) AS total_buy,
        ROUND(AVG(h.times_order * m.price)::numeric, 2) AS avg_spent
    FROM history h
    JOIN meals m ON h.meal_id = m.id
    GROUP BY h.user_id
) stats
WHERE u.id = stats.user_id;