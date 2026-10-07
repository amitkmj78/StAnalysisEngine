-- Lets a user link one Alpaca paper account per portfolio, instead of one
-- total per user. Safe to re-run.

drop index if exists alpaca_paper_accounts_user_idx;
create unique index if not exists alpaca_paper_accounts_user_portfolio_idx
  on alpaca_paper_accounts(user_id, portfolio_id);
