import re

doctopics = open('doc_topics.txt').readlines()[1::]
dois = open('dois_for_topics.txt').readlines()

reordered = []

# Loop over doc_topics lines
for i, line in enumerate(doctopics):
	weights = [0.0] * 160

	# Zip through pairs and create and ordered list of weights
	raw_vals = line.strip().split(' ')[2::]
	pairs = zip(raw_vals[0::2], raw_vals[1::2])
	for p in pairs:
		weights[int(p[0])] = p[1]

	# Calculate doi index
	m = re.search('nidag_(\d+)', line)
	index = int(m.group(1)) - 1

	# Make line
	new_line = dois[index].strip() + '\t' + '\t'.join(weights)
	reordered.append(new_line)

# Write new file with header
outf = open('topics.txt', 'w')
outf.write('Doi\t' + '\t'.join('topic_%d' % (x+1) for x in range(160)) + '\n')
outf.write('\n'.join(reordered))
