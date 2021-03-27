# SAI Project

- EvoCraft environment: need Minecraft account, Java 8
- IEC: sparse user preferences input
- Fitness function learner: neural network classifier (input: minecraft structure, output: % Like/Dislike); *how to train? how often?*
- Population based: N agents, non-concurrent, non-interacting
- CPPN encoding: genome -> CPPN -> Structure generating NN
- NEAT / HyperNEAT evolution: ever-complexifying over evolution
- Minimum Criterion: no 100% air blocks
- Quality diversity: ok from NEAT (topK agents fit to reproduce)
- Divergent: ok from NEAT

What we have:
- NEAT pytorch (with CPPN)
- EvoCraft
- ~ population based

TODO:
[ ] : Define fitness learner model
[ ] : Implement population learning cycle
[ ] : Incorporate "What we have"