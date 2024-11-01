import pandas as pd 
import json

# Create a folder with all the query results:
# {'purpose id': , 'purpose': , 'answer':}
def return_pp_info(row_id):
    pp_id_col = "ID"
    pp_v_col = "Purposes"
    pp_id = int(query_contents.at[row_id, pp_id_col])
    pp_content = query_contents.at[row_id, pp_v_col]
    return pp_id, pp_content

class QExecute:
    def pp1_exe(df:pd.DataFrame):
        page_count = df['page_count']
        res = int(page_count.max())
        return res 

    def pp2_exe(df:pd.DataFrame):
        page_count = df['page_count']
        res = page_count.mean()
        return int(res)

    def pp3_exe(df:pd.DataFrame):
        res = len(df['event'].value_counts())
        return res 

    def pp4_exe(df:pd.DataFrame):
        res = len(df[df['event'] == 'DINNER'])
        return res

    def pp5_exe(df:pd.DataFrame):
        res = len(df[df['event'] == 'LUNCHEON'])
        return res

    def pp6_exe(df:pd.DataFrame):
        res = len(df['venue'].value_counts())
        return res 

    def pp7_exe(df: pd.DataFrame):
        res = len(df[df['occasion'].astype(str).str.lower() == 'daily'])
        return res

    def pp8_exe(df:pd.DataFrame):
        df['occasion'] = df['occasion'].fillna('UNKNOWN')
        res = len(df['occasion'].value_counts())
        return res 

    def pp9_exe(df:pd.DataFrame):
        df['ratio'] = df['dish_count'] / df['page_count']
        # Find the highest ratio
        highest_ratio = df['ratio'].max()
        return highest_ratio

    def pp12_exe(df:pd.DataFrame):
        df['location'] = df['location'].fillna('UNKNOWN')
        filtered_df = df[df['page_count'] > 8]
        # Count the total number of locations in the filtered DataFrame
        res = filtered_df['location'].count()
        return int(res)

    def pp13_exe(df:pd.DataFrame):
        filtered_df = df[df['currency'].str.lower() == 'dollars']
        res = filtered_df['sponsor'].unique().tolist()
        return res

    def pp14_exe(df:pd.DataFrame):
        sponsor_dish_counts = df.groupby('sponsor')['dish_count'].sum()
        # Find the sponsor(s) with the highest number of dishes
        highest_dish_count = sponsor_dish_counts.max()
        top_sponsors = sponsor_dish_counts[sponsor_dish_counts == highest_dish_count].index.tolist()
        return top_sponsors

    def pp15_exe(df):
        breakfast_sponsors = df[df['event'].str.lower() == 'breakfast']['sponsor'].unique().tolist()
        return breakfast_sponsors

    def pp16_exe(df):
        lunch_sponsors = df[df['event'].str.lower() == 'lunch']['sponsor'].unique().tolist()
        return lunch_sponsors

    def pp17_exe(df):
        dinner_sponsors = df[df['event'].str.lower() == 'dinner']['sponsor'].unique().tolist()
        return dinner_sponsors

    def pp18_exe(df):
        sponsor_event_counts = df.groupby('sponsor')['event'].count()
        # Filter sponsors with two or more events
        multiple_events_sponsors = sponsor_event_counts[sponsor_event_counts >= 2].index.tolist()
        return multiple_events_sponsors

    def pp31_exe(df):
        # dataset: chicago 
        unique_risks_count = df['Risk'].nunique()
        return unique_risks_count

    def pp32_exe(df):
        # Standardize the Results column to lowercase
        df['Results'] = df['Results'].str.lower()

        # Total inspections
        total_inspections = len(df)

        # Out-of-business inspections
        out_of_business_inspections = df[df['Results'] == 'out of business'].shape[0]

        # Calculate the percentage of out-of-business inspections
        out_of_business_percentage = (out_of_business_inspections / total_inspections) * 100

        return out_of_business_percentage

    def pp33_exe(df):
        facility_counts = df['Facility Type'].value_counts()
        # Identify the facility type with the most inspections
        # Retrieve the most occurred value
        most_occurred_value = facility_counts.idxmax()
        return most_occurred_value


    def pp34_exe(df):
        facility_counts = df['Facility Type'].value_counts()
        min_count = facility_counts.min()
        least_occurred_values = facility_counts[facility_counts == min_count].index.tolist()
        return least_occurred_values

    def pp36_exe(df):
        failed_inspections_7eleven = df[(df['DBA Name'].str.lower() == '7-eleven'.lower()) & (df['Results'].str.lower() == 'fail')]['Inspection ID'].tolist()
        return failed_inspections_7eleven

    def pp37_exe(df):
        df['Results'] = df['Results'].astype(str).str.lower()
        # Group by DBA_Name and calculate the passing rate
        passing_rate = (
            df.groupby('DBA Name')
            .apply(lambda x: (x['Results'] == 'pass').sum() / len(x))
            .reset_index(name='Passing_Rate')
        )
        best_brand_name = passing_rate.sort_values(by='Passing_Rate', ascending=False).iloc[0]['DBA Name']
        return best_brand_name


    def pp38_exe(df):
        unique_low_risk_facility_types = df[df['Risk'].str.lower() == 'risk 3 (low)']['Facility Type'].unique()
        res = unique_low_risk_facility_types.tolist()
        return res 


    def pp39_exe(df):
        unique_high_risk_facility_types = df[df['Risk'].str.lower() == 'risk 1 (high)']['Facility Type'].unique()
        res = unique_high_risk_facility_types.tolist()
        return res 

    def pp40_exe(df):
        most_frequent_risk_by_type = df.groupby('Facility Type')['Risk'].agg(lambda x: x.value_counts().idxmax()).reset_index().to_json()
        return most_frequent_risk_by_type

    def pp41_exe(df):
        high_risk_facilities = df[df['Risk'].str.contains('risk 1', case=False, na=False)]
        unique_facility_types = high_risk_facilities['Facility Type'].unique().tolist()
        return unique_facility_types

    def pp42_exe(df):
        try:
            high_risk_groceries_count = df[(df['Facility Type'].str.lower() == 'grocery store') & 
                                    (df['Risk'].str.contains('risk 1', case=False))].shape[0]
        except:
            high_risk_groceries_count = df[(df['Facility Type'].astype(str).str.lower() == 'grocery store') & 
                                (df['Risk'].astype(str).str.contains('risk 1', case=False))].shape[0]
        return high_risk_groceries_count

    def pp49_exe(df):
        try:
            safest_school_restaurants_count = df[(df['Facility Type'].str.lower() == 'school') & 
                                        (df['Risk'].str.contains('risk 3', case=False)) & 
                                        (df['Results'].str.lower() == 'pass')].shape[0]
        except:
            safest_school_restaurants_count = df[(df['Facility Type'].astype(str).str.lower() == 'school') & 
                                        (df['Risk'].astype(str).str.contains('risk 3', case=False)) & 
                                        (df['Results'].astype(str).str.lower() == 'pass')].shape[0]
        return safest_school_restaurants_count

    def pp52_exe(df):
        safe_facilities = df[(df['Risk'].str.lower() == 'risk 3 (low)') & (df['Results'].str.lower() == 'pass')]
        safe_addresses = safe_facilities['Address'].tolist()
        return safe_addresses

    def pp62_exe(df):
        avg_res = df['LoanAmount'].mean()
        return avg_res

    def pp63_exe(df):
        highest_loan_amount = df['LoanAmount'].max()
        return highest_loan_amount
    
    def pp64_exe(df):
        lowest_loan_amount = df['LoanAmount'].min()
        return lowest_loan_amount

    def pp65_exe(df):
        """
        pp65: Report all NAICS Codes that indicate job counts greater than 3.0.
        cols: NAICSCode, JobsReported
        """
        result = df[df['JobsReported']> 3]['NAICSCode'].unique()
        return result.tolist()

    def pp66_exe(df):
        """
        pp66: Examine if a correlation exists between jobs reported and the loan amount received.
        cols: JobsReported, LoanAmount
        """
        correlation = df['LoanAmount'].corr(df['JobsReported'])
        return correlation

    def pp67_exe(df):
        """
        pp67: Determine how many PPP loans were disbursed in the city of Honolulu.
        cols: City
        """
        honolulu_loans_count = df[df['City'].str.lower() == 'honolulu'].shape[0]
        return honolulu_loans_count

    def pp68_exe(df):
        """
        pp68: Identify the top 10 business types that received the most PPP loans.
        cols: BusinessType, LoanAmount
        """
        top_business_types = df['BusinessType'].value_counts().head(10)
        # print(top_business_types)
        return top_business_types.index.tolist()

    def pp69_exe(df):
        """
        Identify the types of businesses that received the highest amount of PPP loans and the corresponding dollar amounts.
        cols: BusinessType, LoanAmount
        """
        business_loan_totals = df.groupby('BusinessType')['LoanAmount'].sum().sort_values(ascending=False)
        return business_loan_totals.index[0]
    
    def pp70_exe(df):
        """
        Identify the types of businesses that received the lowest amount of PPP loans and the corresponding dollar amounts.
        cols: BusinessType, LoanAmount
        """
        business_loan_totals = df.groupby('BusinessType')['LoanAmount'].sum().sort_values()
        return business_loan_totals.index[0]

    def pp71_exe(df):
        """
        Check if multiple PPP loans were made to distinct companies sharing the same Zip code.
        cols: BusinessType, Zip
        """
        loans_same_zip_distinct = df.groupby('Zip').filter(lambda x: x['BusinessType'].nunique() > 1 or x['Lender'].nunique() > 1)
        if len(loans_same_zip_distinct) >= 1:
            return "Yes"
        else:
            return "No"

    def pp72_exe(df):
        """
        For a given zip code, determine how many loans were provided.
        cols: Zip, LoanAmount
        """
        loans_per_zip = df.groupby('Zip').size().reset_index(name='LoanCount')
        return loans_per_zip.to_json()
    
    def pp73_exe(df):
        """
        For a given zip code, calculate the total amount of all loans provided.
        cols: Zip, LoanAmount
        """
        total_loan_amount_per_zip = df.groupby('Zip')['LoanAmount'].sum().reset_index(name='TotalLoanAmount')
        return total_loan_amount_per_zip.to_json()

    def pp74_exe(df):
        """
        Which gender type received the most amount of PPP loans and dollar amount? 
        cols: Gende, LoanAmount
        """
        # Calculate total loan count and loan amount by gender
        gender_summary = df.groupby('Gender').agg(
            TotalLoanCount=('LoanAmount', 'size'),
            TotalLoanAmount=('LoanAmount', 'sum')
        ).reset_index()

        # Find the gender type with the most loans and highest loan amount
        most_loans_gender = gender_summary.loc[gender_summary['TotalLoanCount'].idxmax()]
        highest_amount_gender = gender_summary.loc[gender_summary['TotalLoanAmount'].idxmax()]
        return [most_loans_gender['Gender'], highest_amount_gender['Gender']]
    
    def pp75_exe(df):
        """
        Which gender type districts received the least  amount of PPP loans and dollar amount? 
        cols: Gender, LoanAmount
        """
        gender_summary = df.groupby('Gender').agg(
            TotalLoanCount=('LoanAmount', 'size'),
            TotalLoanAmount=('LoanAmount', 'sum')
        ).reset_index()

        # Find the gender type with the least loans and lowest loan amount
        least_loans_gender = gender_summary.loc[gender_summary['TotalLoanCount'].idxmin()]
        lowest_amount_gender = gender_summary.loc[gender_summary['TotalLoanAmount'].idxmin()]

        return [least_loans_gender['Gender'], lowest_amount_gender['Gender']]
    
    def pp76_exe(df):
        """
        Identify the cities that received the highest amount of PPP loans and the corresponding dollar amounts.
        cols: City, LoanAmount
        """
        # Group by City and calculate total loan amounts
        city_loan_totals = df.groupby('City').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()
        print(city_loan_totals)

        # Sort by Total Loan Amount in descending order
        top_cities = city_loan_totals.sort_values(by='TotalLoanAmount', ascending=False)
        print(top_cities['City'][0])
        print(top_cities)
        print(top_cities.iloc[0]['City'])
        # return top_cities['City'][0]
        return top_cities.iloc[0]['City']
    
    def pp77_exe(df):
        """
        Identify the cities that received the lowest amount of PPP loans and the corresponding dollar amounts.
        cols: City, LoanAmount
        """
        city_loan_totals = df.groupby('City').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

        # Sort by Total Loan Amount in ascending order
        lowest_cities = city_loan_totals.sort_values(by='TotalLoanAmount', ascending=True)
        return lowest_cities.iloc[0]['City']
    
    def pp78_exe(df):
        """
        Determine the zip codes that received the highest amount of PPP loans and the corresponding dollar amounts.
        cols: Zip, LoanAmount
        """
        zip_loan_totals = df.groupby('Zip').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

        # Sort by Total Loan Amount in descending order
        highest_zip_codes = zip_loan_totals.sort_values(by='TotalLoanAmount', ascending=False)
        return str(highest_zip_codes.iloc[0]['Zip'])
    
    def pp79_exe(df):
        """
        Determine the zip codes that received the lowest amount of PPP loans and the corresponding dollar amounts.
        cols: Zip, LoanAmount
        """
        zip_loan_totals = df.groupby('Zip').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

        # Sort by Total Loan Amount in descending order
        lowest_zip_codes = zip_loan_totals.sort_values(by='TotalLoanAmount', ascending=True)
        return str(lowest_zip_codes.iloc[0]['Zip'])
    
    def pp80_exe(df):
        """
        Determine which races received the highest amount of PPP loans and the corresponding dollar amounts.
        cols: RaceEthinicity, LoanAmount
        """
        race_loan_totals = df.groupby('RaceEthnicity').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

        # Sort by Total Loan Amount in descending order
        highest_race = race_loan_totals.sort_values(by='TotalLoanAmount', ascending=False)
        return highest_race.iloc[0]['RaceEthnicity']
    
    def pp81_exe(df):
        """
        Determine which races received the lowest amount of PPP loans and the corresponding dollar amounts.
        cols: RaceEthinicity, LoanAmount
        """

        race_loan_totals = df.groupby('RaceEthnicity').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

        # Sort by Total Loan Amount in descending order
        lowest_race = race_loan_totals.sort_values(by='TotalLoanAmount', ascending=True)
        return lowest_race.iloc[0]['RaceEthnicity']
    
    def pp87_exe(df):
        """
        Calculate the ratio of total Loan Amount to Jobs Reported in each city (total Loan Amount in the city divided by total Jobs Reported in the city).
        cols: City, LoanAmount, JobsReported
        """
        # df['City'] = df['City'].str.strip().str.title()  # Standardize city names

        # Group by City and calculate total loan amounts and jobs reported
        city_stats = df.groupby('City').agg(
            TotalLoanAmount=('LoanAmount', 'sum'),
            TotalJobsReported=('JobsReported', 'sum')
        ).reset_index()

        # Calculate the ratio of total Loan Amount to Jobs Reported
        city_stats['LoanToJobRatio'] = city_stats['TotalLoanAmount'] / city_stats['TotalJobsReported']
        return city_stats[['City', 'LoanToJobRatio']].to_json()
    
    def pp89_exe(df):
        """
        Identify geography that have the highest loan amounts. Geography defined by the fields City, State, and Zip Code.
        cols: City, LoanAmount, State, Zip
        """
        # Normalize City names to ensure consistency
        # df['City'] = df['City'].str.strip().str.title()

        # Group by City, State, and Zip, and sum the Loan Amounts
        geo_distribution = df.groupby(['City', 'State', 'Zip']).agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()
        geo_distribution = geo_distribution.loc[geo_distribution['TotalLoanAmount'].idxmax()]
        # geo_distribution = geo_distribution.sort_values(by='TotalLoanAmount', ascending=False)
        return [str(x) for x in geo_distribution[['City', 'State', 'Zip']].tolist()]
    
    def pp92_exe(df):
        """
        Calculate the average number of times each dish has appeared on the menu.
        cols: times_appeared
        """
        average_times_appeared = df['times_appeared'].mean()
        return average_times_appeared

    def pp93_exe(df):
        """
        Identify which dishes have been on the menu for the shortest duration, based on their 'first_appeared' and 'last_appeared' dates.
        cols: first_appeared, last_appeared
        """
        df['last_appeared'] = pd.to_datetime(df['last_appeared'], errors='coerce')
        df['first_appeared'] = pd.to_datetime(df['first_appeared'], errors='coerce')

        # Calculate the difference in years
        df['duration'] = (df['last_appeared'] - df['first_appeared']).dt.days
        # if pd.api.types.is_datetime64_any_dtype(df['last_appeared']) and pd.api.types.is_datetime64_any_dtype(df['first_appeared']):
        #     # Calculate duration as the difference in days if both are datetime
        #     df['duration'] = (df['last_appeared'] - df['first_appeared']).dt.days
        # elif pd.api.types.is_integer_dtype(df['last_appeared']) and pd.api.types.is_integer_dtype(df['first_appeared']):
        #     # Calculate duration as a direct difference if both are integers
        #     df['duration'] = df['last_appeared'] - df['first_appeared']
        # else:
        #     print("Columns are neither both datetime nor both integer.")
        #     df['duration'] = False

        # Filter to identify dishes with the shortest duration
        shortest_duration = df['duration'].min()
        shortest_duration_dishes = df[df['duration'] == shortest_duration]
        return shortest_duration_dishes['name'].tolist() 
        
    
    def pp94_exe(df):
        """
        Identify which dishes have been on the menu for the longest duration, based on their 'first_appeared' and 'last_appeared' dates.
        cols:first_appeared, last_appeared
        """
        try:
            # Calculate duration as a direct difference if both are integers
            df['duration'] = df['last_appeared'] - df['first_appeared']
            longest_duration = df['duration'].max()
            longest_duration_dishes = df[df['duration'] == longest_duration]
            return longest_duration_dishes['name'].tolist()
        except:
            print("Columns are neither both datetime nor both integer.")
            return []
    
    def pp98_exe(df):
        """
        Identify the cheapest dish based on the lowest price.
        cols: name, lowest_price
        """
        df = df[df['lowest_price'] > 0]

        # Find the minimum price in 'lowest_price' column
        min_price = df['lowest_price'].min()

        # Filter the data to find dishes with the lowest price
        cheapest_dish = df[df['lowest_price'] == min_price]
        return cheapest_dish['name'].tolist()
    
    def pp99_exe(df):
        """
        Identify the most expensive dish based on the highest price.
        cols: name, lowest_price
        """
        data = df[df['lowest_price'] > 0]

        # Find the minimum price in 'lowest_price' column
        max_price = data['lowest_price'].max()

        # Filter the data to find dishes with the highest price
        highest_dish = data[data['lowest_price'] == max_price]
        return highest_dish['name'].tolist()

    def pp100_exe(df):
        """
        Find dishes that first appeared before the year 2000.
        cols: name, first_appeared
        """
        dishes_2000 = df[df['first_appeared'] <= 2000]
        earliest_dishes = dishes_2000.loc[dishes_2000['first_appeared'].idxmin()]
        return earliest_dishes['name']

    def pp101_exe(df):
        """
        Identify which dishes were the first to appear on the menu.
        cols: name, first_appeared
        """
        # Find the earliest appearance year
        earliest_year = df['first_appeared'].min()

        # Filter the dishes that appeared in the earliest year
        first_dishes = df[df['first_appeared'] == earliest_year]
        return first_dishes['name'].tolist()
    
    def pp102_exe(df):
        """
        Determine which dishes were the most popular overall on the menus.
        cols:name, menus_appeared
        """
        max_appearances = df['menus_appeared'].max()

        # Filter dishes that have the maximum number of menu appearances
        most_popular_dishes = df[df['menus_appeared'] == max_appearances]
        return most_popular_dishes['name'].tolist()
    
    def pp103_exe(df):
        """
        Determine which dishes were the least popular overall on the menus.
        cols:name, menus_appeared
        """
        min_appearances = df['menus_appeared'].min()

        # Filter dishes that have the minimum number of menu appearances
        least_popular_dishes = df[df['menus_appeared'] == min_appearances]
        return least_popular_dishes['name'].tolist()
    
    def pp104_exe(df):
        """
        Analyze how the highest price has evolved for the top 10 popular dishes, sorting the "times_appeared" column to define the popularity of the dishes.
        cols: name, times_appeared, highest_price
        """
        top_dishes = df.sort_values(by='times_appeared', ascending=False).head(10)
        # return top_dishes[['name', 'highest_price', 'times_appeared']].to_json()
        return {'name': top_dishes['name'].tolist(), 'highest_price': top_dishes['highest_price'].tolist()}

    def pp105_exe(df):
        """
        Analyze how the lowest price has evolved for the top 10 popular dishes, sorting the "times_appeared" column to define the popularity of the dishes.
        cols: name, times_appeared, lowest price
        """
        top_dishes = df.sort_values(by='times_appeared', ascending=True).head(10)
        return {'name': top_dishes['name'].tolist(), 'lowest_price': top_dishes['lowest_price'].tolist()}
    
    def pp106_exe(df):
        """
        Identify which dishes have experienced a lowest price difference.
        cols: name, hightest_price, lowest_price
        """
        df['price_difference'] = df['highest_price'] - df['lowest_price']

        # Identify the dish with the highest price difference
        lowest_price_difference_dish = df.loc[df['price_difference'].idxmin()]
        return lowest_price_difference_dish['name']

    
    def pp107_exe(df):
        """
        Identify which dishes have experienced a highest price difference between highest price and lowest price.
        cols: name, hightest_price, lowest_price
        """    
        try:
            df['price_difference'] = df['highest_price'] - df['lowest_price']

            # Identify the dish with the highest price difference
            highest_price_difference_dish = df.loc[df['price_difference'].idxmax()]
            return highest_price_difference_dish['name']
        except:
            return " "


    def pp108_exe(df):
        """
        Identify the dishes that have the highest average price of a given dish.
        cols: name,  hightest_price, lowest_price
        """
        # Calculate average price for each dish
        df['average_price'] = df[['lowest_price', 'highest_price']].mean(axis=1)

        # Select relevant columns for display
        average_price_comparison = df.loc[df['average_price'].idxmax()]
        return average_price_comparison['name']
    
    def pp109_exe(df):
        """
        Identify the 5 most popular dishes.
        """
        # Sort the dishes by the number of times appeared in descending order
        most_popular_dishes = df.sort_values(by='times_appeared', ascending=False).head(5)

        # Select relevant columns for display
        return most_popular_dishes['name'].tolist()
    
    def pp110_exe(df):
        """
        Identify how the average price has changed for the top 10 most popular dishes, sorting by the "times_appeared" column to assess their popularity. 
        cols: name, times_appeared, highest_price, lowest_price
        """
        # Sort the dishes by the number of times appeared in descending order
        top_10_popular_dishes = df.sort_values(by='times_appeared', ascending=False).head(10)

        top_10_popular_dishes['lowest_price'] = pd.to_numeric(top_10_popular_dishes['lowest_price'], errors='coerce')
        top_10_popular_dishes['highest_price'] = pd.to_numeric(top_10_popular_dishes['highest_price'], errors='coerce')

        # Calculate the average price for each dish
        top_10_popular_dishes['average_price'] = top_10_popular_dishes[['lowest_price', 'highest_price']].mean(axis=1)
        return top_10_popular_dishes[['name', 'average_price']].to_json()


if __name__ == '__main__':
    # Load ground truth dataset 
    gd_parent_folder = "../datasets/ground_truth"
    menu_gd = f"{gd_parent_folder}/menu_all.csv"
    chi_gd = f"{gd_parent_folder}/chi_all.csv"
    ppp_gd = f"{gd_parent_folder}/ppp_all.csv"
    dish_gd = f"{gd_parent_folder}/dish_all.csv"
    menu_df = pd.read_csv(menu_gd)
    chi_df = pd.read_csv(chi_gd)
    ppp_df = pd.read_csv(ppp_gd)
    dish_df = pd.read_csv(dish_gd)
    
    # groundtruth_tag = False
    dirty_tag = False
    groundtruth_tag = False
    qexecute = QExecute
    # Load queries contents
    query_contents = pd.read_csv('../purposes/queries.csv')
    # model = 'dirty'
    # load results by LLMs
    # model = "llama3.1"
    # model = "gemma2"
    model = "mistral"
    model = "llama3.1_1"
    # model = "mistral:7b-instruct"
    # model = "mistral" 
    llm_folder = f"CoT.response/{model}/datasets_llm"
    for query_id in range(0,111):
        row = query_contents[query_contents['ID'] == query_id]

        if len(row) == 0:
            continue
        func = f'pp{query_id}_exe'
        print(func)
        if groundtruth_tag: 
            if query_id >= 62 and query_id <= 91:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/ppp_datasets/cleaned_tables/ppp_sample_p{query_id}.csv'
            elif query_id >= 92:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/dish_datasets/cleaned_tables/dish_sample_p{query_id}.csv'
            elif query_id >= 31 and query_id <= 61:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/chi_food_inspection_datasets/cleaned_tables/chi_sample_p{query_id}.csv'
            elif query_id <31:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/purpose-prepared-datasets/menu/menu_p{query_id}.csv'
        elif model == "llama3.1": 
            if query_id >= 62 and query_id <= 91:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/ppp_test_{query_id}.csv'
            elif query_id >= 92:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/dish_test_{query_id}.csv'
            elif query_id >= 31 and query_id <= 61:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/chi_test_{query_id}.csv'
            elif query_id <31:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/menu_test_{query_id}.csv'
        elif dirty_tag:
            print('dirty data loading...')
            if query_id >= 62 and query_id <= 91:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/ppp_datasets/ppp_data_p{query_id}.csv'
            elif query_id >= 92:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/dish_datasets/dish_data_p{query_id}.csv'
            elif query_id >= 31 and query_id <= 61:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/chi_food_inspection_datasets/chi_food_data_p{query_id}.csv'
            elif query_id <31:
                target_path = f'/projects/bces/lanl2/LLM4DC/datasets/purpose-prepared-datasets/menu/menu_data.csv'
        else:
            if query_id >= 62 and query_id <= 91:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/{model}_ppp_test_{query_id}.csv'
            elif query_id >= 92:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/{model}_dish_test_{query_id}.csv'
            elif query_id >= 31 and query_id <= 61:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/{model}_chi_test_{query_id}.csv'
            elif query_id <31:
                target_path = f'/projects/bces/lanl2/LLM4DC/{llm_folder}/{model}_menu_test_{query_id}.csv'
        print(target_path)
        target_df = pd.read_csv(target_path)
        
        answer = getattr(qexecute, func)(target_df)
        
        print('answer', type(answer), answer)
        result_single = {'pp_id': query_id,
                        'purpose': row['Purposes'].values.tolist()[0],
                        'answer': answer}
        # print(result_single)
        # with open('answer_1-110_dirty.json', 'a') as f:
        #     f.write(json.dumps(result_single))
        #     f.write('\n')
        with open(f'answer_1-110_{model}.json', 'a') as f:
            f.write(json.dumps(result_single))
            f.write('\n')
        
