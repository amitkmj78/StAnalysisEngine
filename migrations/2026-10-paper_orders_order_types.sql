-- BRK-2: stop, stop-limit, trailing-stop and bracket order types on the
-- existing paper-trading ticket -- previously only market/limit existed.
alter table paper_orders add column if not exists order_class text;
alter table paper_orders add column if not exists stop_price real;
alter table paper_orders add column if not exists trail_price real;
alter table paper_orders add column if not exists trail_percent real;
alter table paper_orders add column if not exists take_profit_price real;
alter table paper_orders add column if not exists stop_loss_price real;
alter table paper_orders add column if not exists stop_loss_limit_price real;
