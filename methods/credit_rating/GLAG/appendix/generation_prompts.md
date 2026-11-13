# Generation prompts

The financial-summary generator sends separate system and user messages. The complete message templates are provided as plain text:

- [System message](prompts/generation_system_prompt.txt)
- [User message](prompts/generation_user_prompt.txt)

Replace `{y}` and `{q}` with the reporting year and quarter, and `{input_data}` with that bank's financial facts for the reporting-period file. Preserve the fact identifiers in the input.

The system message requires evidence-grounded analysis organized by CAMELS: Capital, Asset Quality, Management, Earnings, Liquidity, and Sensitivity. The output contains a financial summary, the separator `&&&`, an interbank lending outlook, the separator `|||`, and a comma-separated list of supporting fact identifiers.

See the [Bank B example](supplementary_information.md) for a generated output and its selected input facts.
