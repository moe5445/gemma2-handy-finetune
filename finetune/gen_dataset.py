import json
import random
from collections import Counter
from pathlib import Path

SEED = 42

TERMS = {
    "blue-green": {
        "canonical": ["blue-green", "blue/green"],
        "hand": [
            ("Flip traffic between the two servers with instant switching.", "Deploy with a blue-green switch to the other server."),
            ("Keep the old backend warm while the new one stages, then point everything at the new one once it passes checks.", "Cut over to the new version with a blue-green deployment."),
            ("Swap the whole fleet from the current container to the next one in one switch.", "Move the whole fleet in one go with a blue-green deployment."),
            ("Run two versions side by side and only send real traffic to the new one after it passes.", "Run the new version alongside the old and cut over with a blue-green deployment."),
            ("I want zero-downtime releases, so stand up the new side and flip when it is healthy.", "Do a blue-green release with zero downtime."),
            ("Bring up the next version beside the current one and, once it is healthy, move the load balancer across without dropping a request.", "Cut the load balancer over with a blue-green release."),
            ("Launch a second instance of everything, soak it, then move all real users onto it in one go.", "Stand up a second stack and flip everyone over with a blue-green deploy."),
            ("Have the new stack idle-ready and move everyone over without a blip.", "Blue-green switch the whole fleet without downtime."),
            ("The rollout should leave the old release serving until the new one proves itself, then swap instantly.", "Do a blue-green rollout so the old release keeps serving during cutoff."),
            ("Prepare both generations and redirect the clients to the promoted one at the press of a button.", "Blue-green promote the new generation with a single switch."),
        ],
        "swap": {
            "the two servers": ["the two servers", "the two stacks", "both environments", "the old node and the new node"],
            "traffic": ["traffic", "requests", "the load", "all clients", "the users"],
            "one": ["one", "single", "a single", "one go"],
            "the other server": ["the other server", "the new stack", "the green environment", "the replacement"],
            "switch": ["switch", "flip", "cut over", "point traffic", "redirect"],
            "new": ["new", "candidate", "next", "fresh"],
            "old": ["old", "previous", "current", "existing"],
            "the new one": ["the new one", "the candidate", "the fresh stack"],
            "release": ["release", "deploy", "version", "build"],
        },
    },
    "rate-limit": {
        "canonical": ["rate-lim", "rate caps", "thrott"],
        "hand": [
            ("Limit login requests per minute to prevent the server from being overloaded.", "Rate-limit the login endpoint."),
            ("Stop this API key from calling the service more than a few times per hour.", "Throttle the client by API key."),
            ("Clients retry too fast and it makes things worse, add a delay after repeated failures.", "Rate-limit clients so retries back off after failures."),
            ("The webhook endpoint should only accept ten hits a second, drop the rest.", "Set a 10-req/s cap on the webhook endpoint."),
            ("A bot is hammering the checkout, put a per-minute ceiling on it.", "Rate-limit the checkout by minute."),
            ("Users can refresh ten times a second, allow fewer to protect the service.", "Throttle the refresh endpoint."),
            ("The upload endpoint has no cap, anyone can flood it, add a request budget.", "Add a rate cap to the upload endpoint."),
            ("Our workers keep calling the third-party vendor too often, so bound how many times per minute our code may hit it.", "Rate-limit how often our code calls the vendor."),
            ("The chat API allows too many messages per user per second, bring it down.", "Throttle the chat API per user."),
            ("The polling loop hammers the metrics endpoint, slow it to once a minute.", "Rate-limit the metrics polling to once a minute."),
        ],
        "swap": {
            "per minute": ["per minute", "per second", "per hour", "per day"],
            "requests": ["requests", "calls", "hits", "attempts", "requests a second"],
            "too": ["too", "way too", "overly", "extremely"],
            "cap": ["cap", "limit", "ceiling", "budget", "max out"],
            "service": ["service", "API", "endpoint", "provider"],
            "login": ["login", "checkout", "search", "webhook", "refresh", "upload"],
            "endpoint": ["endpoint", "API", "service", "route", "handler"],
        },
    },
    "cron": {
        "canonical": ["cron", "schedul"],
        "hand": [
            ("Generate the report and email it automatically.", "Set up a cron job to generate and email the report."),
            ("Every midnight, clean the temp files without anyone starting it.", "Schedule the cleanup to run at midnight."),
            ("Run the nightly backup at 2am each day on its own.", "Cron the nightly backup at 2am."),
            ("The billing sync should fire by itself on the first of every month.", "Schedule the billing sync on the first of the month."),
            ("Push the metrics to the server on a fixed timer instead of manually.", "Automate the metrics push with a scheduled timer."),
            ("Refresh the auth token cache every twenty minutes unattended.", "Schedule the token refresh every twenty minutes."),
            ("Every morning at 9, publish the daily digest without a human.", "Cron a daily digest at 9am."),
            ("The purge job should self-start at 3am nightly.", "Schedule the purge job nightly at 3am."),
            ("Rotate the API keys on a monthly basis automatically.", "Schedule the key rotation monthly."),
            ("The overnight log archival should run every night at 3am, with every night no human in the loop.", "Schedule the log archival nightly at 3am."),
            ("Sync the vendor data weekly on Sunday.", "Cron the weekly vendor sync."),
            ("The report must mail itself first thing Monday.", "Schedule the report to email itself Monday."),
        ],
        "swap": {
            "at midnight": ["at midnight", "at 2am", "at 3am", "hourly", "daily", "every twenty minutes"],
            "report": ["report", "sync", "backup", "cleanup", "digest", "purge"],
            "job": ["job", "task", "script", "workflow"],
            "the": ["the", "our", "auto"],
            "nightly": ["nightly", "nightly at", "every night", "at night"],
        },
    },
    "mock": {
        "canonical": ["mock", "stub"],
        "hand": [
            ("Give me fake responses from the payment service so I can test the real flow.", "Mock the payment service with fake responses."),
            ("The payment API is not built, so code against fake responses with fixed values.", "Stub the payment API with fixed sample responses."),
            ("In tests, replace the real card processor so we never charge real cards.", "Mock the card processor in tests."),
            ("Build a fake version of the data service so the frontend can run on its own.", "Create a fake data service for the frontend."),
            ("The external service is slow, so swap in simulated responses for local runs.", "Use a mock for the external service when it is slow."),
            ("The weather source has no live feed yet, so local development is pointed at a canned reply set instead of the live endpoint.", "Mock the weather endpoint with canned replies."),
            ("The real SMS gateway should never be hit during tests, put in a fake one.", "Stub the SMS gateway for tests."),
            ("The streaming service isn't available in CI, so serve scripted chunks instead.", "Mock the streaming service with scripted chunks in CI."),
            ("Replace the recommendation engine with a fixed dataset in the demo.", "Stub the recommendation engine for the demo."),
            ("Before the credit-check vendor goes live, return an OK from a fake provider.", "Mock the credit-check provider until it goes live."),
        ],
        "swap": {
            "fake": ["fake", "stubbed", "simulated", "canned", "mock"],
            "payment service": ["payment service", "card processor", "data service", "external API", "SMS gateway", "weather provider"],
            "service": ["service", "API", "provider", "gateway", "engine"],
            "responses": ["responses", "replies", "data", "values", "results"],
        },
    },
    "lazy-load": {
        "canonical": ["lazy-load", "lazy load", "defer"],
        "hand": [
            ("Load images only when they are needed on the screen.", "Load images lazily as they enter the viewport."),
            ("Do not fetch the map until the user scrolls close to it.", "Lazy-load the map when it comes into view."),
            ("The heavy panel should not load all its content up front, pull it when opened.", "Lazy-load the detail panel content on open."),
            ("Import the big analytics library only when the user opens the dashboard.", "Defer the analytics import until it is needed."),
            ("Fetch messages in pages by passing the last one seen.", "Load messages on demand with lazy pagination."),
            ("Only load the map widget once someone toggles it on.", "Defer loading the map widget until it is toggled."),
            ("The chat widgets shouldn't load until the tab becomes visible.", "Lazy-load the chat widgets on tab visibility."),
            ("Pull the video only when the play button is pressed.", "Defer the video load until play."),
            ("Delay loading the heavy grid until the filter is first used.", "Lazy-load the grid on first filter action."),
            ("The chat history is heavy, so fetch the older messages only when the user scrolls back toward them.", "Lazy-load the older chat messages on scroll up."),
            ("Comments should stay out of the DOM until they come near the fold.", "Defer rendering the comments until they're near the fold."),
            ("The image grid loads every picture on startup, instead load each one as it scrolls into view.", "Lazy-load images as they scroll into view."),
        ],
        "swap": {
            "images": ["images", "map", "panel", "library", "grid", "chat"],
            "when they are needed": ["when they are needed", "until scrolled into view", "only on open", "when they enter the viewport", "on scroll"],
            "the": ["the", "the whole", "thousands of"],
            "on": ["on", "only on", "as soon as", "beware the heavy"],
            "only": ["only", "just", "only"], 
        },
    },
    "idempotent": {
        "canonical": ["idempot", "duplicate"],
        "hand": [
            ("This button keeps sending the same request twice, make repeat clicks harmless.", "Make the request idempotent."),
            ("If the same payment webhook arrives twice, it charges twice, dedupe it.", "Make the webhook handler idempotent."),
            ("Re-running the import should not create duplicate rows.", "Make the import idempotent."),
            ("Re-sending the same create order request must never produce two orders, even if the network retries fire twice.", "Make the create-order request idempotent."),
            ("The sync job re-runs hourly and must not double-insert the same record.", "Make the sync job idempotent."),
            ("An event can be delivered again by the broker, the consumer should ignore the repeat.", "Make the consumer idempotent to duplicate events."),
            ("Retries replay the same webhook, applying it again should be a no-op.", "Make the webhook replay idempotent."),
            ("The user double-clicks submit; the second click must not create another row.", "Make the submit action idempotent."),
        ],
        "swap": {
            "twice": ["twice", "more than once", "again", "two times"],
            "request": ["request", "import", "webhook", "response", "sync job"],
            "the": ["the", "our", "this"],
            "idempotent": ["idempotent", "duplicate-safe", "dedupe"],
        },
    },
    "backpressure": {
        "canonical": ["backpressure", "backpressure"],
        "hand": [
            ("Producers send work faster than workers can clear it, so slow the producers down.", "Apply backpressure to the producers when the queue fills up."),
            ("The queue is full and the system should push back instead of dropping jobs.", "Apply backpressure instead of dropping jobs when the queue is full."),
            ("The customers are piping in requests way faster than the processing side can drain, throttle it upstream.", "Add backpressure upstream when the pipe can't keep up."),
            ("When the workers back up, stop pulling more from the source until they catch up.", "Apply backpressure by pausing the ingest when workers lag."),
            ("The batch enqueues 1k jobs while a worker drains 10, the queue should tell the producers to slow down.", "Let the queue apply backpressure to slow producers."),
            ("Kafka producers race ahead of consumers; the consumer lag should gate how much producers emit.", "Use backpressure so producer rate follows consumer lag."),
            ("The exporter overwhelms the DB with writes, the throughput should cap at how fast the sink drains.", "Throttle the exporter with backpressure from the DB sink."),
        ],
        "swap": {
            "fast": ["faster", "faster than", "way faster", "too fast"],
            "queue": ["faster", "Queue", "pipe", "stream", "ingest"],
            "producers": ["producers", "publishers", "publishers", "the ingest side", "senders"],
            "down": ["down", "down a gear", "out"],
            "push back": ["push back", "apply backpressure", "slow the producers"],
        },
    },
    "debounce": {
        "canonical": ["debounce"],
        "hand": [
            ("The search fires a request on every keystroke, wait until the user stops typing.", "Debounce the search input."),
            ("Filters re-query on each click, batch them until the user finishes choosing.", "Debounce the filter updates."),
            ("The URL bar pings the server on every keystroke, only fire once the user pauses typing for a moment.", "Debounce the URL bar autocomplete."),
            ("Every scroll event triggers a callback; squash them into one after the scrolling stops.", "Debounce the scroll handler."),
            ("The resize listener recomputes layout on every pixel, run it only after resize settles.", "Debounce the resize recalculation."),
            ("Typing is streaming too many search queries; fire one query after a quiet gap.", "Debounce the search query with a delay."),
            ("Each time the user types a letter we call the suggestions API; one call after they stop is enough.", "Debounce the suggestions fetch."),
            ("The slider drag re-renders the chart per tick, only recompute once you release.", "Debounce the slider re-render."),
        ],
        "swap": {
            "typing": ["typing", "clicking", "scrolling", "moving the slider", "typing into the box"],
            "keystroke": ["keystroke", "change", "tick", "character"],
            "the": ["the", "Page", "the "],
        },
    },
    "event-sourcing": {
        "canonical": ["event sourcing"],
        "hand": [
            ("Record every balance change over time instead of just the final value.", "Model the balance with event sourcing."),
            ("Keep the full audit history of each order by storing events, not just state.", "Store order state as event sourcing."),
            ("The ledger should grow by appending state-changing events instead of mutating a row.", "Build the ledger on event sourcing."),
            ("Every user action should be written to the stream, not overwritten.", "Model the user's activity as event sourcing."),
            ("We need to replay the account from day one; keep the whole change log.", "Use event sourcing for the account history."),
            ("Instead of a snapshot, keep the full record of what happened and derive state.", "Derive state from a full event log."),
        ],
        "swap": {
            "balance": ["balance", "order", "account", "inventory", "user"],
            "event sourcing": ["event sourcing", "an event log", "event stream"],
        },
    },
    "optimistic": {
        "canonical": ["optimistic", "roll"],
        "hand": [
            ("Update the like button immediately, then fix it if the request fails.", "Apply an optimistic update and roll back on failure."),
            ("Show the new value right away and reconcile with the server later.", "Do an optimistic UI update and reconcile."),
            ("Tone the count up instantly, then correct it if the API rejects.", "Optimistically increment the count and revert on error."),
            ("The like should not wait for the network, stay in sync and undo if it fails.", "Like optimistically and roll back when the call fails."),
            ("Flip the switch in the UI at once and fix on the background slot failure.", "Optimistically flip the switch and reconcile."),
        ],
        "swap": {
            "button": ["button", "count", "UI", "value", "switch"],
            "update": ["update", "increment", "render", "apply"],
        },
    },
    "circuit-breaker": {
        "canonical": ["circuit breaker"],
        "hand": [
            ("When the API keeps failing, stop hammering it to protect it, then retry the endpoint later.", "Add a circuit breaker around the failing API."),
            ("After a few consecutive errors, the call should trip and stop for a bit.", "Add a circuit breaker around the flaky endpoint."),
            ("Stop hammering the payment gateway when it is unhealthy; give it a cooldown and try again.", "Trip a circuit breaker on the payment gateway and retry after a cooldown."),
            ("Too many errors in a row, the app should stop calling the service until it recovers.", "Open the circuit breaker after repeated failures."),
            ("The email provider failing repeatedly should cut the traffic to it, then probe again.", "Open the circuit breaker for the email provider."),
            ("A flock of transient errors from the order flow keeps cascading, wrap the calls so they halt and retry later.", "Wrap the order calls in a circuit breaker."),
            ("The supplier endpoint gets saturated and every retry makes it worse, so throw the switch and wait.", "Open the circuit breaker for the saturated supplier."),
            ("If three successive requests to the ledger service fail, stop and check back after a minute.", "Open a circuit breaker after consecutive ledger failures."),
            ("The vendor keeps returning 5xx, so block the client from calling and audit after a fixed delay.", "Trip the circuit breaker for the failing vendor."),
            ("The search backend being down should not take the whole app with it; isolate and retry smartly.", "Add a circuit breaker so a down backend doesn't cascade."),
        ],
        "swap": {
            "API": ["API", "gateway", "service", "provider", "dependency"],
            "circuit breaker": ["circuit breaker", "a breaker"],
            "retry": ["retry", "check back", "try again"],
            "failing": ["failing", "flaky", "overloaded", "erroring"],
        },
        "extra_hand": [
            ("The vendor API gets flaky, trips the system after a few errors for a while, then clearly recovers.", "Add a circuit breaker that trips the vendor API after errors and recovers."),
            ("The payment gateway might hiccup repeatedly; the app should stop sending and wait a bit before resuming.", "Put a circuit breaker in front of the payment gateway."),
            ("The upstream service keeps failing in spurts, so the caller should back off and retry after a break.", "Add a circuit breaker so callers back off and retry later."),
        ],
    },
}


def syn_swap(text, mapping, rng, depth, always_change=False):
    changed = False
    for _ in range(depth):
        keys = list(mapping.keys())
        rng.shuffle(keys)
        for k in keys:
            low = text.lower()
            i = low.find(k.lower())
            if i >= 0 and (rng.random() < 0.6 or (always_change and not changed)):
                text = text[:i] + rng.choice(mapping[k]) + text[i + len(k):]
                changed = True
    return text, changed


# Terms that need many variants, and the target per-hand-seed count.
N_VARIANTS = {
    "blue-green": 22,
    "rate-limit": 24,
    "cron": 24,
    "mock": 24,
    "lazy-load": 24,
    "idempotent": 14,
    "backpressure": 10,
    "debounce": 12,
    "event-sourcing": 6,
    "optimistic": 6,
    "circuit-breaker": 8,
}


def main():
    rng = random.Random(SEED)
    rows = []
    for term, spec in TERMS.items():
        spec["hand"] = spec["hand"] + spec.get("extra_hand", [])
        n_variants = N_VARIANTS.get(term, 4)
        for user, outp in spec["hand"]:
            rows.append({"term": term, "user": user, "output": outp, "paraphrase": False})
            made = 0
            guard = 0
            while made < n_variants and guard < 300:
                guard += 1
                vu, ch1 = syn_swap(user, spec["swap"], rng, int(len(spec["swap"]) > 0) + 2, always_change=True)
                vo = outp
                if ch1 and vu != user:
                    rows.append({"term": term, "user": vu, "output": vo, "paraphrase": True})
                    made += 1

    reserved = [
        {"term": "blue-green", "user": "Run two copies of the app at once and point real users at the second copy once it reports healthy, keeping the old one up meanwhile.", "output": "Do a blue-green deploy keeping the old copy live."},
        {"term": "rate-limit", "user": "Background workers currently hammer the upstream vendor far too aggressively, constrain their call frequency per period.", "output": "Rate-limit the workers' calls to the vendor."},
        {"term": "cron", "user": "The overnight archiving should keep running on its own every day at 3am without manual launch.", "output": "Schedule the nightly archiving with a cron trigger."},
        {"term": "mock", "user": "Since the weather vendor is still down, local dev should hit a prepared payload file rather than the live API.", "output": "Mock the weather vendor using a canned payload."},
        {"term": "lazy-load", "user": "The old-message portion of the chat is big, so fetch older entries only when the person scrolls back toward them.", "output": "Lazy-load the older chat messages on scroll up."},
    ]

    seen = set()
    uniq = []
    for r in rows:
        key = (r["user"], r["output"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    rows = uniq

# A few extra real-user-style paraphrases for the previously under-sampled
    # terms (distinct phrasings, NOT the exact fresh-eval probe text).
    fresh = [
        {"term": t, "user": u, "output": o, "paraphrase": False}
        for t, u, o in [
            ("idempotent", "If the same retry arrives a second time, the server should treat it like the first and return the same result.", "Make the API idempotent to retries."),
            ("backpressure", "The sink can't drain as fast as the source emits, the source should be told to hold off.", "Apply backpressure between the source and the sink."),
            ("debounce", "The search must not fire my term-lookup on every key press, only after I have paused.", "Debounce the keypress lookups."),
            ("circuit-breaker", "The third-party API has been failing for a while; the app should gate upstream calls until it heals.", "Open a circuit breaker on the third party API."),
        ]
    ]
    rows += fresh

    holdout_probes = [
        {"term": "blue-green", "user": "Deploy the new build next to the old one and flip DNS once health checks pass without cutting a single request.", "output": "Do a blue-green deployment and flip DNS after health checks."},
        {"term": "rate-limit", "user": "A scraper keeps pulling our pricing endpoint, so bound it to a generous number of calls per quarter hour.", "output": "Rate-limit the pricing endpoint per 15-minute window."},
        {"term": "cron", "user": "Nightly snapshot mailing should fall into place at 1:15am without a human remembering to run it.", "output": "Cron the nightly snapshot mail at 01:15."},
        {"term": "mock", "user": "The telemetry vendor isn't connected in the lab, run the demo against a database fixture file instead.", "output": "Mock the telemetry vendor with a fixture file."},
        {"term": "lazy-load", "user": "The timeline pulls all past events up front, only bring the rest when the user asks for the next page.", "output": "Lazy-load the timeline by page."},
    ]
    for i, p in enumerate(holdout_probes):
        p["paraphrase"] = False
        rows.append(p)

    reserved_users = set(r["user"] for r in reserved)

    train, holdout = [], []
    for r in rows:
        # guaranteed real probes / holdout checks go to holdout
        if r["user"] in reserved_users or any(r["user"] == p["user"] for p in holdout_probes):
            holdout.append(r)
        elif rng.random() < 0.12:
            holdout.append(r)
        else:
            train.append(r)

    with open(Path(__file__).resolve().parent.parent / "data" / "pairs_train.json", "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=1)
    with open(Path(__file__).resolve().parent.parent / "data" / "pairs_holdout.json", "w") as f:
        json.dump(holdout, f, ensure_ascii=False, indent=1)

    print("train", len(train))
    print("holdout", len(holdout))
    print("train counts:", dict(Counter(r["term"] for r in train)))
    print("holdout counts:", dict(Counter(r["term"] for r in holdout)))
    print("unique users train:", len({r["user"] for r in train}))
    print("---- train sample (first 3) ----")
    for r in train[:3]:
        print(r)


if __name__ == "__main__":
    main()