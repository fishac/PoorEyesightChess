import pandas as pd
import sys

def main():
	if len(sys.argv) >= 2:
		if sys.argv[1] == 'all':
			max_entries = int(sys.argv[1])
			parsed_data = pd.read_csv('./board_states.csv')
			parsed_data.to_csv('./board_states_trimmed_shuffled.csv',index=False)
		else:
			max_entries = int(sys.argv[1])
			parsed_data = pd.read_csv('./board_states.csv')
			parsed_data[:max_entries].sample(frac=1).to_csv('./board_states_trimmed_shuffled.csv',index=False)
	else:
		print('Requires one command line argument: number of rows to keep. Or the word \'all\' without quotes.')

if __name__ == '__main__':
	main()
	
