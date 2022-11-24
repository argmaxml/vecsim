# vecsim - an interface for similarity servers
A standard, light-weight interface to all popular similarity servers.

## The problems we are trying to solve:
1. **Standard API** - Different vector similarity servers has different APIs - so switching is not trivial.
1. **Identifiers** - Some vector similiarity servers support string IDs, some do not - we keep track of the mapping.
1. **Partitions** - In most cases, a pre-filtering is needed prior to querying, we abstract this concept away.

## Use cases for similarity servers
1. **Recommendation**
1. **Lookalikes**
1. **Search**
1. **Anomaly Detection**