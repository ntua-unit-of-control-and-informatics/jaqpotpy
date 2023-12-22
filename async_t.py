# df = pd.read_csv('/Users/pantelispanka/Desktop/gdp-countries.csv')
#
#
# y = df['GDP']
# X = df[['LFG', 'EQP', 'NEQ', 'GAP']]
#
# loop = asyncio.get_event_loop()
# a = loop.create_task(jha.calculate_a(X))
# b = loop.create_task(jha.calculate_doa_matrix(X))
# all_groups = asyncio.gather(a, b)
# results = loop.run_until_complete(all_groups)
# print(results[0])
# print(results[1])
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.debug('This message should appear on the console')
logging.info('So should this')
logging.warning('And this, too')



