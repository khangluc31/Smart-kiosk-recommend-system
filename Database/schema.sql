
DROP TABLE IF EXISTS rec_logs, cart_items, order_items, orders, sessions,
                     history, user_fav_cat, user_fav_meal,
                     meals_item, option_groups, items, meals, categories, users CASCADE;
CREATE TABLE users (
    id          uuid PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE categories (
    id    serial PRIMARY KEY,
    name  text NOT NULL UNIQUE
);

CREATE TABLE meals (
    id           serial PRIMARY KEY,
    source_id    text UNIQUE,              
    cat_id       int NOT NULL REFERENCES categories(id),
    name         text NOT NULL,
    description  text,                        
    price        numeric(10,0) NOT NULL CHECK (price >= 0),
    available    bool NOT NULL DEFAULT true,
    is_hotsell   bool NOT NULL DEFAULT false,
    is_bestsell  bool NOT NULL DEFAULT false,
    is_seasonal  bool NOT NULL DEFAULT false,
    is_combo     bool NOT NULL DEFAULT false
);

CREATE TABLE items (
    id    serial PRIMARY KEY,
    name  text NOT NULL UNIQUE
);

CREATE TABLE option_groups (
    id           serial PRIMARY KEY,
    meal_id      int NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    name         text NOT NULL,
    min_choices  int NOT NULL DEFAULT 1,
    max_choices  int NOT NULL DEFAULT 1,
    CHECK (min_choices >= 0 AND min_choices <= max_choices)
);

CREATE TABLE meals_item (
    id           serial PRIMARY KEY,
    meal_id      int NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    item_id      int NOT NULL REFERENCES items(id),
    quantity     int NOT NULL DEFAULT 1 CHECK (quantity > 0),
    group_id     int REFERENCES option_groups(id) ON DELETE CASCADE,
    extra_price  numeric(10,0) NOT NULL DEFAULT 0,  
    UNIQUE NULLS NOT DISTINCT (meal_id, item_id, group_id)
);

CREATE TABLE sessions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid REFERENCES users(id),          
    started_at  timestamptz NOT NULL DEFAULT now(),
    transcript  text,                                
    budget      numeric(10,0)                        
);

CREATE TABLE orders (
    id          serial PRIMARY KEY,
    session_id  uuid NOT NULL REFERENCES sessions(id),
    user_id     uuid REFERENCES users(id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    total       numeric(10,0) NOT NULL CHECK (total >= 0)
);
CREATE INDEX orders_user_time_idx ON orders (user_id, created_at);

CREATE TABLE order_items (
    id          serial PRIMARY KEY,
    order_id    int NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    meal_id     int NOT NULL REFERENCES meals(id),
    quantity    int NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit_price  numeric(10,0) NOT NULL                 -- giá lúc mua
);
CREATE INDEX order_items_order_idx ON order_items (order_id);
CREATE INDEX order_items_meal_idx  ON order_items (meal_id);

CREATE TABLE cart_items (
    id          serial PRIMARY KEY,
    session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    meal_id     int NOT NULL REFERENCES meals(id),
    quantity    int NOT NULL DEFAULT 1 CHECK (quantity > 0)
);


CREATE TABLE rec_logs (
    id          serial PRIMARY KEY,
    session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    meal_id     int NOT NULL REFERENCES meals(id),
    stage       text NOT NULL CHECK (stage IN ('initial', 'upsell')),
    source      text NOT NULL CHECK (source IN ('linucb', 'two_tower', 'item2vec', 'ilp')),
    position    int NOT NULL,
    score       double precision,
    chosen      bool NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX rec_logs_session_idx ON rec_logs (session_id);
CREATE VIEW user_stats AS
SELECT user_id,
       COUNT(*)              AS num_buy,
       ROUND(AVG(total))     AS avg_spent,
       MAX(created_at)       AS last_order_at
FROM orders
WHERE user_id IS NOT NULL
GROUP BY user_id;

CREATE VIEW user_meal_counts AS
SELECT o.user_id,
       oi.meal_id,
       SUM(oi.quantity)       AS times_order,   
       COUNT(DISTINCT o.id)   AS num_orders,    
       MAX(o.created_at)      AS last_buy
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE o.user_id IS NOT NULL
GROUP BY o.user_id, oi.meal_id;

CREATE VIEW user_fav_meal AS
SELECT user_id, meal_id, num_orders, rnk
FROM (
    SELECT umc.*,
           ROW_NUMBER() OVER (PARTITION BY user_id
                              ORDER BY num_orders DESC, last_buy DESC) AS rnk
    FROM user_meal_counts umc
) t
WHERE rnk <= 5;

CREATE VIEW user_fav_cat AS
SELECT user_id, cat_id, num_orders, rnk
FROM (
    SELECT umc.user_id,
           m.cat_id,
           SUM(umc.num_orders) AS num_orders,
           ROW_NUMBER() OVER (PARTITION BY umc.user_id
                              ORDER BY SUM(umc.num_orders) DESC) AS rnk
    FROM user_meal_counts umc
    JOIN meals m ON m.id = umc.meal_id
    GROUP BY umc.user_id, m.cat_id
) t
WHERE rnk <= 3;

CREATE VIEW meal_stats AS
SELECT m.id                                   AS meal_id,
       COALESCE(SUM(oi.quantity), 0)          AS num_sold,
       COALESCE(SUM(oi.quantity) FILTER (
           WHERE o.created_at >= now() - INTERVAL '30 days'), 0) AS num_sold_30d
FROM meals m
LEFT JOIN order_items oi ON oi.meal_id = m.id
LEFT JOIN orders o       ON o.id = oi.order_id
GROUP BY m.id;