# Supplementary information examples

We present selected supplementary facts collected from the SEC--EDGAR API for three representative banking institutions in 2023Q3. Each line reports the fact identifier, simplified financial concept, reporting period, and value; the ellipses indicate additional facts omitted for brevity. We further provide the LLM-generated output for Bank B, including its financial summary, interbank lending outlook, and returned reference facts. Bold identifiers correspond to the facts displayed in the example.

## Bank A

| Fact ID | Concept | Reporting period | Value |
| --- | --- | --- | --- |
| f114  | Revenues | 2023-07-01--2023-09-30 | 25,167,000,000 USD |
| f174  | EarningsPerShareBasic | 2023-07-01--2023-09-30 | 0.91 USD/share |
| f178  | EarningsPerShareDiluted | 2023-07-01--2023-09-30 | 0.90 USD/share |
| f674  | BrokerageCommissionsRevenue | 2023-07-01--2023-09-30 | 3,963,000,000 USD |
| f690  | InvestmentBankingRevenue | 2023-07-01--2023-09-30 | 1,188,000,000 USD |
| f118  | FinancingAndOffBalanceSheetCreditLossProvisionReversal | 2023-07-01--2023-09-30 | 1,234,000,000 USD |
| f6067 | GainsLossesOnLoansHeldForSale | 2023-07-01--2023-09-30 | -28,000,000 USD |
| f4410 | SharesReservedForFutureIssuance | 2023-09-30 | 496,000,000 shares |
| f5103 | Level3AssetSales | 2023-07-01--2023-09-30 | 303,000,000 USD |
| f5105 | Level3AssetSettlements | 2023-07-01--2023-09-30 | 82,000,000 USD |

[Download selected facts](examples/bank_a_2023Q3_facts.txt)

## Bank B

| Fact ID | Concept | Reporting period | Value |
| --- | --- | --- | --- |
| f63   | InvestmentBankingRevenue | 2023-07-01--2023-09-30 | 1,722,000,000 USD |
| f64   | InvestmentBankingRevenue | 2022-07-01--2022-09-30 | 1,674,000,000 USD |
| f67   | PrincipalTransactionsRevenue | 2023-07-01--2023-09-30 | 6,210,000,000 USD |
| f68   | PrincipalTransactionsRevenue | 2022-07-01--2022-09-30 | 5,383,000,000 USD |
| f121  | ProvisionForLoanLeaseAndOtherLosses | 2023-01-01--2023-09-30 | 6,558,000,000 USD |
| f122  | ProvisionForLoanLeaseAndOtherLosses | 2022-01-01--2022-09-30 | 4,101,000,000 USD |
| f171  | EarningsPerShareBasic | 2023-07-01--2023-09-30 | 4.33 USD/share |
| f172  | EarningsPerShareBasic | 2022-07-01--2022-09-30 | 3.13 USD/share |
| f7415 | TierOneRiskBasedCapital | 2023-09-30 | 268,579,000,000 USD |
| f7416 | TierOneRiskBasedCapital | 2023-09-30 | 282,482,000,000 USD |

[Download selected facts](examples/bank_b_2023Q3_facts.txt)

## Bank C

| Fact ID | Concept | Reporting period | Value |
| --- | --- | --- | --- |
| f100  | Revenues | 2023-07-01--2023-09-30 | 5,233,000,000 USD |
| f160  | EarningsPerShareBasic | 2023-07-01--2023-09-30 | 3.60 USD/share |
| f164  | EarningsPerShareDiluted | 2023-07-01--2023-09-30 | 3.60 USD/share |
| f3170 | Level3AssetSales | 2023-07-01--2023-09-30 | 186,000,000 USD |
| f3172 | Level3AssetSettlements | 2023-07-01--2023-09-30 | 233,000,000 USD |
| f4010 | DerivativeAssetSetoffRightAmount | 2023-09-30 | 1,396,000,000 USD |
| f4247 | CustomerContractRevenueExcludingTax | 2023-07-01--2023-09-30 | 5,233,000,000 USD |
| f2409 | LongTermDebtPrincipalDueWithinOneYear | 2023-09-30 | 14,193,000,000 USD |
| f558  | CumulativeDerivativeLossOnHTMDebtSecurities | 2023-09-30 | 297,000,000 USD |

[Download selected facts](examples/bank_c_2023Q3_facts.txt)

## LLM-generated output for Bank B

[Download the original output](examples/bank_b_2023Q3_output.txt)

### Financial status summary

The bank demonstrates robust financial performance in 2023Q3, driven by strong growth in principal transactions and investment banking revenue. Principal transactions revenue surged to \$6.21 billion, a significant increase from \$5.38 billion in the same quarter of the prior year, while investment banking revenue also grew modestly to \$1.72 billion from \$1.67 billion. This top-line strength translated into a substantial earnings per share (EPS) of \$4.33, up from \$3.13 in 2022Q3. However, this performance is counterbalanced by a notable deterioration in asset quality, as evidenced by a full-year provision for credit losses of \$6.56 billion, which is substantially higher than the \$4.10 billion recorded in the first nine months of 2022, reflecting a more cautious outlook on loan portfolio risk. The bank’s capital position remains solid, with Tier 1 risk-based capital of approximately \$269-\$282 billion as of September 2023, well above regulatory minimums, including a stress capital buffer of 4% and a capital conservation buffer of 2.5%. The balance sheet shows a large and stable base of debt securities, though these are marked by significant unrealized losses, indicating sensitivity to interest rate movements.

### Interbank lending outlook

The bank's strong capital adequacy and profitability suggest it is a reliable counterparty in the interbank market, making it likely that other institutions would be willing to lend to it if needed. Its ability to generate substantial trading and principal transaction revenues indicates a sophisticated and active trading book, which often correlates with a role as a net lender in the interbank market. However, the sharply increased provisioning for loan losses signals heightened credit risk within its core lending activities. While this does not immediately impair its liquidity or capital, it introduces an element of caution. Consequently, while the bank is likely to continue participating as a lender in the interbank network, its lending activity might be tempered by a more conservative internal risk appetite, and counterparties may scrutinize its credit more closely despite its strong capital ratios.

### Reference facts

**f63**, **f64**, **f67**, **f68**, f119, f120, **f121**, **f122**, **f171**, **f172**, f239, f240, f245, f246, f253, f254, f295, f296, f315, f316, **f7415**, **f7416**, f7419, f7420, f7378, f7379, f7387, f7388

Bold identifiers occur in the selected Bank B facts above; the full reference list is retained in the original output.
