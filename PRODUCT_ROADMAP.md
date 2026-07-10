# AI Vendor Intelligence Platform — Product Roadmap

**Created:** July 8, 2026  
**Current Status:** MVP Demo → SaaS Product  
**Goal:** Make something users would pay for

---

## Executive Summary

Current product is a working **demo** but lacks:
- Real financial data (scores feel arbitrary)
- Historical tracking (one-off briefs only)
- Multi-user support (no authentication)
- Cost visibility (unknown expense per brief)
- Data reliability (news collection hits LLM limits at 33/50 companies)

**3-month plan to turn this into a paid SaaS product.**

---

## Phase 1: Data & Cost Foundation (Weeks 1-3)

### 1.1 Financial Data Integration
**Problem:** Briefs say "FINANCIAL HEALTH: 70/100" with no actual financial metrics

**Solution:**
- Integrate Crunchbase API for funding data
  - Last funding round amount & date
  - Current valuation
  - Investor lists
  - Estimated burn rate
- Integrate AngelList for additional startup metrics
- Integrate LinkedIn API for headcount tracking

**Outcome:**
```
FINANCIAL HEALTH: 70/100 [MEDIUM CONFIDENCE]
- Last funding: Series C, $100M (Jan 2024)
- Valuation: $1.2B (down from $1.5B)
- Burn rate: $2M/month (18 months runway)
- Employees: 280 (↑15% YoY)
- CEO: John Doe (tenure: 3 years)

Sources: Crunchbase (Jan 2024), LinkedIn (Jan 2025)
```

**Effort:** 3-4 hours
**Cost:** $100-200/month API fees (or free tier)
**Impact:** Scores now credible, not guesses

### 1.2 Cost Monitoring & Optimization
**Problem:** No visibility into cost per brief

**Solution:**
- Langfuse already tracks tokens
- Calculate cost: `(input_tokens * $0.00015) + (output_tokens * $0.0006)`
- Log to database: `{company, cost_cents, date, model}`
- Build simple dashboard: cost trends, most expensive companies

**Target Cost:** <$0.10 per brief  
**Current Cost:** ~$0.30-0.50 per brief (estimate)

**Effort:** 2 hours
**Impact:** Understand profitability, find optimization opportunities

### 1.3 Prompt Engineering & Score Calibration
**Problem:** Scores feel arbitrary, generic synthesis text

**Solution:**
- Create scoring rubric per dimension:
  ```
  Financial Health 80+: Series B+, 24+ months runway, 20%+ YoY growth
  Financial Health 60-79: Series A, 12-23 months runway, slowing growth
  Financial Health <60: Pre-seed/Seed, <12 months runway, cash burn
  ```
- Make scores explainable:
  ```
  Financial Health: 70/100
  Reasoning:
    ✓ Recent funding: $50M Series B → +25 pts
    ✓ 18-month runway → +20 pts
    ✗ No IPO path → -15 pts
    ✗ Headcount down 10% → -10 pts
    ? Revenue not disclosed → 0 pts
  ```
- Test on 10 known companies:
  - OpenAI (should be 95) → Measure actual
  - Anthropic (should be 92) → Measure actual
  - Unknown startup (should be 35) → Measure actual
- A/B test: old prompts vs new, measure user preference

**Effort:** 2-3 days (iterative)
**Impact:** Briefs shift from "feels made up" to "actually credible"

---

## Phase 2: Fix Data Bottlenecks (Weeks 4-6)

### 2.1 News Collection Bottleneck
**Problem:** Groq free tier = 100K tokens/day → hits limit at company #33

**Solution Options (Pick one or combine):**

**Option A: Rule-based + LLM fallback (RECOMMENDED)**
- Use keyword matching for 80% of news
  ```python
  if "acquisition" in headline: score += 30
  if "layoffs" in headline: score -= 20
  if "funding" in headline: score += 25
  if "lawsuit" in headline: score -= 15
  ```
- Only send ambiguous cases to LLM
- Result: 80% accuracy, 10% of cost
- Cost: ~$10/month

**Option B: Batch processing**
- Collect RSS feeds daily (free)
- Classify only on Sundays (spread cost over 7 days)
- Cost: 100K tokens / 7 = ~14K tokens/day available

**Option C: Paid tier**
- Groq Pro: $20/month for 1M tokens/day
- Or Anthropic API: $3 per 1M tokens
- Cost: $50-100/month

**Recommendation:** Start with Option A (fastest to implement, cheapest)

**Effort:** 1 day
**Cost Savings:** $0.20-0.30 per brief
**Impact:** Scales to 50 companies without hitting limits

### 2.2 Improve Data Coverage
**Problem:** Some companies have no EDGAR, low GitHub activity, missing data

**Solution:**
- Add alternative data sources:
  - Hunter.io → Company email patterns (hiring signals)
  - Clearbit → Company growth metrics
  - Glassdoor API → Employee reviews, salary trends
- Add confidence scoring per dimension:
  ```
  Financial Health: UNKNOWN [no public funding data]
  Technology Momentum: 65/100 [based on 3 GitHub repos only]
  ```

**Effort:** 1-2 days
**Impact:** More complete briefs, better confidence scores

---

## Phase 3: Historical Tracking & Trends (Weeks 7-8)

### 3.1 Daily Scheduled Collection
**Problem:** Users run one search, lose context. No trend detection.

**Solution:**
- Schedule daily brief generation for top 50 companies
  ```python
  # Runs nightly at 2 AM
  for company in TOP_50:
      brief = generate_brief(company)
      store_brief(company, date.today(), brief)
  ```
- Track metrics over time:
  ```
  Cohere Financial Health:
  - Jan 1: 65/100
  - Jan 8: 67/100 (↑ funding news)
  - Jan 15: 70/100 (↑ revenue announcement)
  ```
- Detect anomalies:
  - "OpenAI score dropped 15 points" → Alert
  - "Unknown startup jumped to 80/100" → Flag

**Effort:** 1 day
**Impact:** Users see trends, understand what changed

### 3.2 Comparison & Leaderboards
**Solution:**
- Leaderboards:
  - "Top 10 by Financial Health"
  - "Fastest growing (Technology Momentum)"
  - "Most stable (Personnel Stability)"
- Side-by-side comparison:
  - Search "Cohere" → Show similar companies (Mistral, Together AI)
  - Comparison table with all metrics

**Effort:** 2 days
**Impact:** Competitive benchmarking becomes possible

---

## Phase 4: Authentication & SaaS Setup (Weeks 9-10)

### 4.1 User Accounts
**Solution:**
- Simple auth: SQLite + JWT tokens
- Sign up with email
- Save favorite companies (5-10)
- Email alerts: "Cohere funding news" (weekly digest)

**Effort:** 2 days
**Cost:** Free (build yourself)

### 4.2 Pricing & Launch
**Freemium Model:**

| Tier | Price | Features |
|------|-------|----------|
| **Free** | $0 | 5 one-time searches/month |
| **Pro** | $29/month | Track 20 companies, alerts, 2-year history |
| **Enterprise** | $500+/month | Unlimited tracking, real-time alerts, API access |

**Effort:** 1 day
**Impact:** First revenue

---

## Phase 5: Polish & Quality (Weeks 11-12, Ongoing)

### 5.1 Citation & Source Quality
- Every claim must cite source:
  - "Series B: $50M" → Crunchbase link + date
  - "GitHub activity up 30%" → GitHub URL + date range
  - "Lost CTO" → LinkedIn URL

- Remove generic filler:
  ```
  ❌ BAD: "This is an AI company doing machine learning"
  ✅ GOOD: "Specializes in LLM fine-tuning for enterprises, 15+ public HF models"
  ```

### 5.2 Weekly Prompt Optimization
- A/B test prompts on known companies
- Measure: "Are users saying this is accurate?"
- Iterate based on feedback

---

## Database Optimization (Do This First!)

### Cleanup Script
```python
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()

# Remove duplicates (keep latest)
cursor.execute("""
    DELETE FROM ai_company_signals
    WHERE id IN (
        SELECT id FROM (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY company_name, headline 
                       ORDER BY created_at DESC
                   ) as rn
            FROM ai_company_signals
        ) t
        WHERE rn > 1
    )
""")
print(f"✅ Deleted {cursor.rowcount} duplicates")

# Delete signals older than 90 days
cursor.execute("""
    DELETE FROM ai_company_signals
    WHERE created_at < NOW() - INTERVAL '90 days'
""")
print(f"✅ Deleted {cursor.rowcount} old signals")

conn.commit()
cursor.close()
conn.close()
```

### Caching Strategy

**Level 1: Brief Result Cache (24 hours)**
```python
# Cache directory: cache/briefs/{company_name}.txt
# Check cache before running agents
# If <24 hours old, return cached brief
# Result: 2nd request = instant, no cost
```

**Level 2: Signal Query Cache (6 hours)**
```python
# Use Redis or LRU cache
# Cache funding_signals, github_activity, etc.
# TTL: 6 hours
# Result: Don't re-query DB for same company within 6 hours
```

**Level 3: Frontend Cache (localStorage)**
- Already implemented
- Keep as-is

---

## Success Metrics

| Phase | Goal | Success Metric |
|-------|------|----------------|
| **Phase 1** | Data & Cost | Cost <$0.10/brief, financial data in briefs, citable |
| **Phase 2** | Reliability | Cover 50 companies daily without limits, confidence scores |
| **Phase 3** | Tracking | Dashboard shows trends, users set 2-3 alerts |
| **Phase 4** | SaaS | 50 beta users, $500/month revenue |
| **Phase 5** | Quality | NPS >40, users say "saves 5 hours/week" |

---

## Critical Path (12 weeks total)

```
Week 1-3:    Phase 1 (Data + Cost + Prompts)
Week 4-6:    Phase 2 (Fix bottlenecks)
Week 7-8:    Phase 3 (Historical tracking)
Week 9-10:   Phase 4 (Auth + SaaS)
Week 11-12:  Phase 5 (Polish)
```

---

## Quick Wins (Start This Week)

1. **Database cleanup** (5 min)
   - Run dedup + delete old data script
   - Database shrinks 20-30%

2. **Cost tracking** (2 hours)
   - Query Langfuse for token costs
   - Log to database
   - See which companies are expensive

3. **Add Crunchbase API** (3 hours)
   - Most impactful for credibility
   - Makes financial scores real

4. **Fix news bottleneck** (4 hours)
   - Implement rule-based + LLM fallback
   - Scales to 50 companies

---

## Resources to Research

- **Crunchbase API:** https://www.crunchbase.com/api
- **AngelList API:** https://www.angellist.com/api
- **Redis caching:** https://redis.io/ (Python: `pip install redis`)
- **Prompt engineering:** OpenAI Cookbook (applies to any LLM)
- **SaaS pricing:** https://www.priceintelligently.com

---

## Notes for Future Self

- Don't overthink architecture. Ship features first.
- Talk to 5 potential users before Phase 4 (auth). Validate demand.
- Track cost/brief obsessively. It's your unit economics.
- Prompts improve iteratively. Test on known companies weekly.
- Financial data is the credibility multiplier. Get this right.

---

**Next Step:** Pick one item from Phase 1 and start this week.  
**Questions?** Check LANGFUSE_SETUP.md for observability setup.
