"""
generate_dataset.py

Builds a synthetic-but-realistic dataset of (incoming email, sent reply) pairs
that a small company's support/sales inbox might see, plus a small
human-labeled *calibration* set used later to validate the evaluation metric.

WHY SYNTHETIC + TEMPLATED (see README "Dataset" section for the full rationale):
- No real customer email corpus is available to us without violating someone's
  privacy, and public email corpora (Enron, etc.) are personal/legal
  correspondence, not customer-support Q&A, so they don't match the
  suggested-reply use case.
- Templates + randomized slots (names, order numbers, products, dates, tone)
  give us control over label quality (we KNOW the reply is a good, on-policy
  answer to the email) while still producing lexical variety, so the dataset
  is honest about being synthetic but is representative of the *structure* of
  real support/sales correspondence: a request/question + constraints, and a
  reply that acknowledges, answers, and gives next steps.

Categories covered (chosen to span the common shapes of business email):
  1. order_status        - "where is my order"
  2. refund_request       - customer wants money back
  3. bug_report           - technical issue report
  4. sales_inquiry        - prospect asking about pricing/features
  5. meeting_scheduling   - propose/confirm a meeting time
  6. billing_question     - invoice / subscription charge question
  7. cancellation_request - cancel subscription/order
  8. complaint_feedback   - unhappy customer, general complaint

Run:
    python3 generate_dataset.py
Outputs:
    emails_dataset.jsonl          (train/retrieval corpus + held-out test split)
    human_eval_calibration.jsonl  (hand-labeled quality scores, for metric validation)
"""
import json
import random
from pathlib import Path

random.seed(42)

FIRST_NAMES = ["Priya", "Daniel", "Wei", "Fatima", "Carlos", "Emma", "Kenji", "Sofia",
               "Liam", "Aisha", "Noah", "Yuki", "Mateo", "Grace", "Omar", "Anya"]
LAST_NAMES = ["Sharma", "Kim", "Novak", "Rossi", "Mueller", "Diallo", "Tanaka", "Ivanov",
              "Costa", "Park", "Nguyen", "Bianchi"]
PRODUCTS = ["the Pro Plan subscription", "the Starter Kit", "the wireless keyboard (model K200)",
            "the annual analytics license", "the noise-cancelling headphones",
            "the team workspace add-on", "the express shipping upgrade", "the API access tier"]
COMPANIES = ["Northwind Supplies", "Brightloop", "Cedarline Goods", "Quanta Analytics",
             "Fernhill Devices", "Marlowe & Co"]

def name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def order_num():
    return f"#{random.randint(10000, 99999)}"

def date_str():
    day = random.randint(1, 28)
    month = random.choice(["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"])
    return f"{month} {day}"

AGENT_SIGNOFFS = ["Best,\nAlex from Support", "Warm regards,\nJordan, Customer Care",
                  "Thanks,\nSam @ Support Team", "Kind regards,\nMorgan, Customer Success"]

def signoff():
    return random.choice(AGENT_SIGNOFFS)

# ---------- Category generators ----------
# Each returns (subject, incoming_body, reply_body, tags:list[str], key_points:list[str])
# key_points = the concrete asks/questions in the incoming email that a *good* reply
# must address. Used later by the evaluator for "coverage" scoring.

def gen_order_status(rng):
    n, num, prod, d = name(), order_num(), random.choice(PRODUCTS), date_str()
    late = rng.choice([True, False])
    subject = f"Where is my order {num}?"
    body = (f"Hi,\n\nI placed an order for {prod} on {d} (order {num}) and it still says "
            f"\"processing.\" Could you tell me the current status and when it will ship? "
            f"{'This is later than the estimate I was given.' if late else ''}\n\n{n}")
    ship_date = date_str()
    reply = (f"Hi {n.split()[0]},\n\nThanks for reaching out, and sorry for the uncertainty. "
             f"I checked order {num} and it's confirmed and moving to fulfillment — the current "
             f"expected ship date is {ship_date}, with tracking sent to this email once it leaves "
             f"the warehouse. "
             f"{'I also want to apologize for the delay past the original estimate; ' + rng.choice(['I have applied a 10% credit to your account.', 'I have flagged this with our warehouse team to prevent it happening again.'])} "
             f"Let me know if you don't see tracking by then and I'll dig in further.\n\n{signoff()}")
    return subject, body, reply, ["order_status"], ["current status of order", "expected ship/delivery date"]

def gen_refund_request(rng):
    n, num, prod = name(), order_num(), random.choice(PRODUCTS)
    reason = rng.choice(["it arrived damaged", "it's not compatible with my setup",
                          "I ordered it by mistake", "it doesn't match the description"])
    subject = f"Refund request for order {num}"
    body = (f"Hello,\n\nI'd like to request a refund for {prod} (order {num}) because {reason}. "
            f"Can you let me know the process and how long it takes?\n\nThanks,\n{n}")
    eligible = rng.choice([True, True, False])
    if eligible:
        reply = (f"Hi {n.split()[0]},\n\nI'm sorry to hear that — thanks for flagging it. You're "
                 f"within our 30-day return window, so I've started a refund for order {num}. "
                 f"You'll get a prepaid return label by email; once the item is scanned by the "
                 f"carrier, the refund posts to your original payment method within 5-7 business "
                 f"days. No need to wait for us to receive it before the label goes out.\n\n{signoff()}")
    else:
        reply = (f"Hi {n.split()[0]},\n\nThanks for reaching out, and I'm sorry for the frustration. "
                 f"Order {num} is just outside our standard 30-day return window, so I'm not able "
                 f"to process a full refund automatically. That said, I can offer a store credit "
                 f"for the full amount, or escalate to my manager for a case-by-case exception if "
                 f"you'd like — just let me know which you'd prefer.\n\n{signoff()}")
    return subject, body, reply, ["refund_request"], ["refund eligibility/process", "expected timeline"]

def gen_bug_report(rng):
    n = name()
    feature = rng.choice(["the CSV export", "two-factor login", "the mobile app sync",
                           "the dashboard charts", "the calendar integration"])
    subject = f"{feature} is broken"
    body = (f"Hi team,\n\n{feature} has stopped working for me since yesterday — "
            f"I get an error/blank screen when I try to use it. I've tried refreshing and "
            f"logging out and back in. Is this a known issue, and is there a fix or workaround?\n\n{n}")
    reply = (f"Hi {n.split()[0]},\n\nThanks for the detailed report. "
             f"{rng.choice(['This is a known issue our engineering team is actively fixing and should be resolved within 24 hours.', 'I was not able to reproduce this immediately, so I have opened a ticket and our engineers are investigating.'])} "
             f"In the meantime, a workaround is to {rng.choice(['use the desktop site instead of the app', 'clear your browser cache and try an incognito window', 'try again after a hard refresh (Ctrl+Shift+R)'])}. "
             f"I'll follow up personally as soon as it's fixed either way.\n\n{signoff()}")
    return subject, body, reply, ["bug_report"], ["acknowledgement of the specific bug", "workaround or fix timeline"]

def gen_sales_inquiry(rng):
    n, comp = name(), random.choice(COMPANIES)
    size = rng.choice(["a team of 8", "about 40 people", "a small team of 3", "150+ employees"])
    subject = "Pricing question before we commit"
    body = (f"Hi,\n\nWe're {comp} and evaluating your product for {size}. Could you share your "
            f"pricing tiers and whether there's a discount for annual billing? Also, do you "
            f"support SSO?\n\n{n}")
    reply = (f"Hi {n.split()[0]},\n\nGreat to hear from you! For a team of your size, our Team "
             f"plan is the best fit at $29/user/month billed monthly, or $24/user/month if billed "
             f"annually (about a 17% saving). "
             f"{'SSO (SAML) is included on the Team plan.' if rng.random() > 0.3 else 'SSO (SAML) is available as an add-on on the Team plan, or included on Enterprise.'} "
             f"Happy to set up a 20-minute call this week to answer any other questions — "
             f"would Tuesday or Wednesday afternoon work?\n\n{signoff()}")
    return subject, body, reply, ["sales_inquiry"], ["pricing tiers", "annual discount", "SSO support"]

def gen_meeting_scheduling(rng):
    n = name()
    topic = rng.choice(["the Q3 roadmap", "onboarding your team", "the integration walkthrough",
                         "contract renewal"])
    day1, day2 = rng.sample(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 2)
    subject = f"Scheduling a call about {topic}"
    body = (f"Hi,\n\nCould we set up 30 minutes to discuss {topic}? I'm generally free "
            f"{day1} or {day2} afternoon this week — let me know what works on your end.\n\n{n}")
    reply = (f"Hi {n.split()[0]},\n\n{day1} afternoon works well on my end — does 2:00pm your "
             f"time work? I'll send a calendar invite with a video link for {topic} once you "
             f"confirm. If that time doesn't work, {day2} afternoon is my backup.\n\n{signoff()}")
    return subject, body, reply, ["meeting_scheduling"], ["proposed a specific day/time", "confirmed the meeting topic"]

def gen_billing_question(rng):
    n = name()
    amt = f"${rng.randint(15,300)}.{rng.randint(0,99):02d}"
    subject = "Unexpected charge on my account"
    body = (f"Hello,\n\nI noticed a charge of {amt} on my card from you that I don't recognize. "
            f"Could you tell me what this is for and confirm my current plan?\n\n{n}")
    prod_phrase = random.choice(PRODUCTS)
    prod_phrase = prod_phrase[4:] if prod_phrase.startswith("the ") else prod_phrase
    reply = (f"Hi {n.split()[0]},\n\nThanks for flagging this — I checked your account and the "
             f"{amt} charge is from your {prod_phrase}, billed on your renewal date. "
             f"Your current plan is active and in good standing. If you believe this was billed "
             f"in error or you'd like to change your plan, I'm happy to help — just let me know.\n\n{signoff()}")
    return subject, body, reply, ["billing_question"], ["explanation of the charge", "confirmation of current plan"]

def gen_cancellation(rng):
    n = name()
    reason = rng.choice(["we're switching tools", "budget cuts", "we no longer need it", "it wasn't the right fit"])
    subject = "Requesting to cancel my subscription"
    body = (f"Hi,\n\nI'd like to cancel my subscription — {reason}. Could you confirm the "
            f"cancellation and let me know if I'll be charged again?\n\n{n}")
    reply = (f"Hi {n.split()[0]},\n\nSorry to see you go, and thanks for letting us know. I've "
             f"processed the cancellation — you won't be billed again, and you'll keep access "
             f"until the end of your current billing period. "
             f"{'If you don' + chr(39) + 't mind sharing more about what didn' + chr(39) + 't work, it really helps us improve.' if rng.random()>0.5 else 'If anything changes, we would love to have you back.'}\n\n{signoff()}")
    return subject, body, reply, ["cancellation_request"], ["confirmation of cancellation", "whether they will be charged again"]

def gen_complaint(rng):
    n = name()
    issue = rng.choice(["was on hold for 40 minutes and never got an answer",
                         "received the wrong item twice in a row",
                         "have emailed three times with no response",
                         "was told conflicting information by two different agents"])
    subject = "Very frustrated with recent support experience"
    body = (f"Hi,\n\nI {issue}. I've been a customer for two years and this is not the "
            f"experience I expect. Can someone please help resolve this?\n\n{n}")
    reply = (f"Hi {n.split()[0]},\n\nI'm really sorry — that's not the experience we want you to "
             f"have, especially as a two-year customer, and I understand the frustration. "
             f"I'm personally taking ownership of this now: {rng.choice(['I have corrected the issue on your account and added a credit for the trouble.', 'I am escalating this to our team lead so it does not happen again.'])} "
             f"You can reach me directly at this email if anything else comes up.\n\n{signoff()}")
    return subject, body, reply, ["complaint_feedback"], ["acknowledgement/apology", "concrete resolution or next step"]

GENERATORS = [gen_order_status, gen_refund_request, gen_bug_report, gen_sales_inquiry,
              gen_meeting_scheduling, gen_billing_question, gen_cancellation, gen_complaint]

def build_main_dataset(n_per_category=15):
    rng = random.Random(7)
    records = []
    idx = 0
    for gen in GENERATORS:
        for _ in range(n_per_category):
            subject, body, reply, tags, key_points = gen(rng)
            idx += 1
            records.append({
                "id": f"email_{idx:04d}",
                "category": tags[0],
                "subject": subject,
                "incoming_email": body,
                "sent_reply": reply,
                "key_points": key_points,
            })
    rng.shuffle(records)
    # 85/15 split: retrieval corpus (what the generator learns from) vs held-out test set
    split = int(len(records) * 0.85)
    for i, r in enumerate(records):
        r["split"] = "corpus" if i < split else "test"
    return records

def build_calibration_set():
    """
    Small hand-authored set of (email, candidate reply, human_score) used ONLY to
    validate that our automatic metric correlates with human judgment of quality.
    Scores are 1 (bad) - 5 (excellent), assigned by us acting as the human rater,
    with an explicit rationale so the labels are auditable, not arbitrary.
    """
    calib = [
        {
            "email": "Hi, I placed an order for the wireless keyboard (model K200) on March 4 "
                     "(order #48213) and it still says \"processing.\" Could you tell me the "
                     "current status and when it will ship?\n\nDaniel Kim",
            "reply": "Hi Daniel,\n\nThanks for reaching out. I checked order #48213 — it's "
                     "confirmed and moving to fulfillment, with an expected ship date of April 2. "
                     "Tracking will be emailed once it leaves the warehouse. Let me know if you "
                     "don't see it by then.\n\nBest,\nAlex from Support",
            "key_points": ["current status of order", "expected ship/delivery date"],
            "human_score": 5,
            "human_rationale": "Directly answers both questions (status + ship date), right tone, offers a clear next step."
        },
        {
            "email": "Hi, I placed an order for the wireless keyboard (model K200) on March 4 "
                     "(order #48213) and it still says \"processing.\" Could you tell me the "
                     "current status and when it will ship?\n\nDaniel Kim",
            "reply": "Hello, thanks for your email. We are looking into it.",
            "key_points": ["current status of order", "expected ship/delivery date"],
            "human_score": 1,
            "human_rationale": "Vague, doesn't state status or a ship date, no real information given."
        },
        {
            "email": "Hi, I placed an order for the wireless keyboard (model K200) on March 4 "
                     "(order #48213) and it still says \"processing.\" Could you tell me the "
                     "current status and when it will ship?\n\nDaniel Kim",
            "reply": "Hi Daniel, thanks for your patience. Your order is confirmed and being "
                     "processed. We don't have a ship date yet but we'll notify you.\n\nBest,\nAlex",
            "key_points": ["current status of order", "expected ship/delivery date"],
            "human_score": 3,
            "human_rationale": "Answers status but not the ship-date question with any specificity; polite but incomplete."
        },
        {
            "email": "I'd like to request a refund for the noise-cancelling headphones (order "
                     "#77291) because it arrived damaged. Can you let me know the process and how "
                     "long it takes?\n\nThanks,\nSofia Rossi",
            "reply": "Hi Sofia,\n\nI'm sorry to hear that. You're within our 30-day return window, "
                     "so I've started a refund for order #77291 — a prepaid label is on its way, "
                     "and the refund posts within 5-7 business days after the carrier scans it.\n\n"
                     "Warm regards,\nJordan, Customer Care",
            "key_points": ["refund eligibility/process", "expected timeline"],
            "human_score": 5,
            "human_rationale": "Covers eligibility, process, and timeline; empathetic and specific."
        },
        {
            "email": "I'd like to request a refund for the noise-cancelling headphones (order "
                     "#77291) because it arrived damaged. Can you let me know the process and how "
                     "long it takes?\n\nThanks,\nSofia Rossi",
            "reply": "We can look into a refund. Please send us more photos of the box.",
            "key_points": ["refund eligibility/process", "expected timeline"],
            "human_score": 2,
            "human_rationale": "Doesn't confirm eligibility or timeline; adds a new ask without explaining why, feels evasive."
        },
        {
            "email": "We're Brightloop and evaluating your product for a team of 8. Could you "
                     "share your pricing tiers and whether there's a discount for annual billing? "
                     "Also, do you support SSO?\n\nEmma Costa",
            "reply": "Hi Emma,\n\nFor a team of 8, our Team plan is $29/user/month monthly, or "
                     "$24/user/month annually (~17% savings). SSO (SAML) is included on Team. "
                     "Happy to do a quick call this week if useful.\n\nThanks,\nSam @ Support Team",
            "key_points": ["pricing tiers", "annual discount", "SSO support"],
            "human_score": 5,
            "human_rationale": "All three questions (pricing, discount, SSO) answered precisely, with a helpful offer."
        },
        {
            "email": "We're Brightloop and evaluating your product for a team of 8. Could you "
                     "share your pricing tiers and whether there's a discount for annual billing? "
                     "Also, do you support SSO?\n\nEmma Costa",
            "reply": "Thanks for reaching out! We have great plans for teams of all sizes. Let me "
                     "know if you want to hop on a call.",
            "key_points": ["pricing tiers", "annual discount", "SSO support"],
            "human_score": 1,
            "human_rationale": "Answers none of the three specific questions asked; pure filler."
        },
        {
            "email": "I've been on hold for 40 minutes and never got an answer. I've been a "
                     "customer for two years and this is not the experience I expect. Can someone "
                     "please help resolve this?\n\nWei Kim",
            "reply": "Hi Wei,\n\nI'm really sorry — that's not the experience we want you to have. "
                     "I'm personally taking ownership now and have escalated this to my team lead "
                     "so it doesn't happen again. You can reach me directly at this email.\n\n"
                     "Kind regards,\nMorgan, Customer Success",
            "key_points": ["acknowledgement/apology", "concrete resolution or next step"],
            "human_score": 5,
            "human_rationale": "Apologizes, takes ownership, gives a concrete next step and a direct contact."
        },
        {
            "email": "I've been on hold for 40 minutes and never got an answer. I've been a "
                     "customer for two years and this is not the experience I expect. Can someone "
                     "please help resolve this?\n\nWei Kim",
            "reply": "Sorry about that! Is there anything else I can help you with today?",
            "key_points": ["acknowledgement/apology", "concrete resolution or next step"],
            "human_score": 2,
            "human_rationale": "Generic apology, doesn't acknowledge the specific complaint or offer any resolution."
        },
        {
            "email": "I'd like to cancel my subscription — we're switching tools. Could you "
                     "confirm the cancellation and let me know if I'll be charged again?\n\n"
                     "Aisha Diallo",
            "reply": "Hi Aisha,\n\nSorry to see you go. I've processed the cancellation — you "
                     "won't be billed again, and you'll keep access until the end of your current "
                     "billing period.\n\nBest,\nAlex from Support",
            "key_points": ["confirmation of cancellation", "whether they will be charged again"],
            "human_score": 5,
            "human_rationale": "Confirms both things asked: cancellation done, and no further charges."
        },
        {
            "email": "I'd like to cancel my subscription — we're switching tools. Could you "
                     "confirm the cancellation and let me know if I'll be charged again?\n\n"
                     "Aisha Diallo",
            "reply": "We're sad to see you go! Your feedback matters to us.",
            "key_points": ["confirmation of cancellation", "whether they will be charged again"],
            "human_score": 1,
            "human_rationale": "Never actually confirms the cancellation or billing question — the two things asked."
        },
        {
            "email": "Could we set up 30 minutes to discuss the Q3 roadmap? I'm generally free "
                     "Monday or Tuesday afternoon this week.\n\nKenji Tanaka",
            "reply": "Hi Kenji,\n\nMonday afternoon works well — does 2:00pm your time work? I'll "
                     "send a calendar invite once you confirm. Tuesday afternoon is my backup if "
                     "not.\n\nThanks,\nSam @ Support Team",
            "key_points": ["proposed a specific day/time", "confirmed the meeting topic"],
            "human_score": 5,
            "human_rationale": "Proposes a specific day/time and confirms topic implicitly via the invite offer."
        },
        {
            "email": "Could we set up 30 minutes to discuss the Q3 roadmap? I'm generally free "
                     "Monday or Tuesday afternoon this week.\n\nKenji Tanaka",
            "reply": "Sure, let's chat sometime soon!",
            "key_points": ["proposed a specific day/time", "confirmed the meeting topic"],
            "human_score": 1,
            "human_rationale": "No day/time proposed at all despite two options being given."
        },
    ]
    for i, c in enumerate(calib):
        c["id"] = f"calib_{i:03d}"
    return calib

if __name__ == "__main__":
    out_dir = Path(__file__).parent
    main_ds = build_main_dataset(n_per_category=15)
    with open(out_dir / "emails_dataset.jsonl", "w") as f:
        for r in main_ds:
            f.write(json.dumps(r) + "\n")
    calib = build_calibration_set()
    with open(out_dir / "human_eval_calibration.jsonl", "w") as f:
        for r in calib:
            f.write(json.dumps(r) + "\n")
    n_corpus = sum(1 for r in main_ds if r["split"] == "corpus")
    n_test = sum(1 for r in main_ds if r["split"] == "test")
    print(f"Wrote {len(main_ds)} records ({n_corpus} corpus / {n_test} test) to emails_dataset.jsonl")
    print(f"Wrote {len(calib)} records to human_eval_calibration.jsonl")
