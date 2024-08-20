def get_examples_from_df(df, n_examples):

	# collect used examples to exclude from test dataset
	indexes_to_drop = []

	# collect examples with single fault
	examples = {}

	for c in df.columns[1:]:
		if df[c].sum() == 0: continue
		examples[c] = []
		i = df.loc[ (df[c] == 1) & (df[df.columns[1:][df.columns[1:]!=c]].sum(axis=1) == 0) ].sample(n_examples)
		indexes_to_drop.append(i.index)
		for e in i.values:
			examples[c].append([e[0], "[" + ",".join(map(str, e[1:])) + "]"])
			
	# collect examples with multiple faults
	examples_multiple = {}

	t = df.loc[df[df.columns[1:]].sum(axis=1) == 2]
	idx = t.sample(n_examples).index
	indexes_to_drop.append(idx)
	t = df.iloc[idx, :].values

	for r in t:
		c = "&".join(df.columns[1:][r[1:]==1])
		if examples_multiple.get(c) is None:
			examples_multiple[c] = []
		examples_multiple[c].append([r[0], "[" + ",".join(map(str, r[1:])) + "]"])
		
	# add both single and multiple into one place
	examples.update(examples_multiple)

	return indexes_to_drop, examples