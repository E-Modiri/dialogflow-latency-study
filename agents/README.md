# Dialogflow ES agent exports

These are the four Dialogflow ES agents used in the revised experiments (5–7 November 2026), exported after the runs. The Weather, Currency and Joke agents are unchanged since the runs; for the Capital City agent, two unused intents were removed after the runs (see below). Each folder is the agent's export; each `.zip` can be restored with Dialogflow console → ⚙ → Export and Import → Restore from ZIP.

| Folder | Domain | Task intent | Parameter (entity) | Training phrases |
|---|---|---|---|---|
| `WeatherAgent` | Weather | `How is the weather?` | `geo-city` (`@sys.geo-city`, required) | 5 |
| `CapitalCityAgent` | Capital City | `GetCapital` | `country` (`@sys.geo-country`) | 4 |
| `CurrencyAgent_corrected` | Currency | `ConvertCurrency` | `from_currency`, `to_currency` (`@sys.currency-name`) | 12 |
| `JokeAgent` | Joke | `GetJoke` | `category` (`@sys.any`) | 11 |

All agents use English, Dialogflow ES Trial edition, global location, ML classification threshold 0.3, and webhook fulfillment on the task intent only.

## Redactions

- **Webhook URLs:** each agent's fulfillment URL was replaced with `https://YOUR-TUNNEL-HOST/webhook/<domain>`. Set it to the URL printed by `bench_harness.py` before running.

Nothing else was changed.

## Original Currency agent (before correction)

The Currency agent used in the original submission had this configuration, which extracted the correct currency pair for **0 of 60** test queries (`data/currency_extraction/currency_extraction_original_agent.csv`):

- `from_currency`: `@sys.unit-currency` (matches only an amount followed by a currency, e.g. "1 GBP")
- `to_currency`: `@sys.currency-name`, *Is list* enabled
- Two training phrases: "How much is 1 GBP in INR?" and "Convert 1 USD to EUR"

The corrected agent (`CurrencyAgent_corrected`) extracts all 60 pairs correctly (`data/currency_extraction/currency_extraction_corrected_agent.csv`). Its training phrases use currency pairs and wording that differ from the test queries.

## Known properties of the measured agents

Apart from the Capital City change noted below, these agents are released as measured, including the following properties:

- **Overlap with test queries:** three of the four Capital City training phrases (Japan, France, Canada) and two of the eleven Joke training phrases ("Give me a dark joke", "Tell me a programming joke") are identical to test queries. Weather and Currency training phrases do not overlap with the test set. The study measures latency, not NLU accuracy, so this does not affect the reported results, but it should be considered by anyone reusing the agents for accuracy evaluation.
- **Annotation slips:** in the Weather agent, "How is the weather?" annotates the word *weather* as an unused `color` parameter, and one phrase annotates *France* as a city. In the Joke agent, one phrase annotates *and* as the category.
- **Extra intents during measurement:** when the experiments were run, the Capital City agent also contained two intents left over from development (`GetCurrency` and `ordering pizza`), neither with webhook fulfillment. They were deleted after the measurements, before this export, because one of them contained an API key in a response text. The task intent `GetCapital`, its four training phrases, its parameter and the agent settings are unchanged. The agent also contains a small unused custom entity (`country`), and the Weather agent contains one unused custom entity (`geo_sys`).
