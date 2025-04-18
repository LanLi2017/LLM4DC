import pandas as pd 
import json
import numpy as np
import re
import os

# Create a folder with all the query results:
# {'purpose id': , 'purpose': , 'answer':}
def return_pp_info(row_id):
    pp_id_col = "ID"
    pp_v_col = "Purposes"
    pp_id = int(query_contents.at[row_id, pp_id_col])
    pp_content = query_contents.at[row_id, pp_v_col]
    return pp_id, pp_content

ISO_8601_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

def safe_parse_datetime(value):
    """Only parse values that match the ISO 8601 format; return original if not."""
    return pd.to_datetime(value, errors='coerce') if ISO_8601_REGEX.match(str(value)) else value

class QExecute:
    def pp1_exe(df:pd.DataFrame):
        page_count = df['page_count']
        res = int(page_count.max())
        return res 

    def pp2_exe(df:pd.DataFrame):
        page_count = df['page_count']
        res = page_count.mean()
        return int(res)

    def pp3_exe(df: pd.DataFrame):
        """
        Return a list of unique event values.
        cols: event
        """
        try:
            # Extract unique event values and return as a list
            return df['event'].dropna().unique().tolist()
        
        except Exception as e:
            return []


    def pp4_exe(df:pd.DataFrame):
        res = len(df[df['event'] == 'DINNER'])
        return res

    def pp5_exe(df:pd.DataFrame):
        res = len(df[df['event'] == 'LUNCHEON'])
        return res

    def pp6_exe(df: pd.DataFrame):
        """
        Return a list of unique venue values.
        cols: venue
        """
        try:
            # Extract unique venue values and return as a list
            return df['venue'].dropna().unique().tolist()
        
        except Exception as e:
            return []


    def pp7_exe(df: pd.DataFrame):
        res = len(df[df['occasion'].astype(str).str.lower() == 'daily'])
        return res

    def pp8_exe(df: pd.DataFrame):
        """
        Return a list of unique occasion values, replacing NaN with 'UNKNOWN'.
        cols: occasion
        """
        try:
            # Extract unique occasion values and return as a list
            return df['occasion'].dropna().unique().tolist()
        
        except Exception as e:
            return []


    def pp9_exe(df:pd.DataFrame):
        df['ratio'] = df['dish_count'] / df['page_count']
        # Find the highest ratio
        highest_ratio = df['ratio'].max()
        return highest_ratio
    
    def pp10_exe(df:pd.DataFrame):
        try:
            df['dish_count'] = pd.to_numeric(df['dish_count'], errors='coerce')
            clean_df = df.dropna(subset=['dish_count'])
            avg_dish_count = clean_df['dish_count'].mean()
            return int(avg_dish_count) if not np.isnan(avg_dish_count) else None
        except Exception as e:
            return None

    def pp11_exe(df:pd.DataFrame):
        try:
            # Convert columns to numeric, coercing errors to NaN
            df['dish_count'] = pd.to_numeric(df['dish_count'], errors='coerce')
            df['page_count'] = pd.to_numeric(df['page_count'], errors='coerce')

            # Drop rows where conversion failed (NaNs appear due to dirty data)
            df = df.dropna(subset=['dish_count', 'page_count'])

            # Compute the ratio
            df['ratio'] = df['dish_count'] / df['page_count']

            # Get the highest ratio
            highest_ratio = df['ratio'].max()

            # Return None if no valid ratio exists
            return highest_ratio if not np.isnan(highest_ratio) else None

        except Exception as e:
            print(f"Error: {e}")
            return None

    def pp12_exe(df: pd.DataFrame):
        """
        Return a list of unique locations where page_count > 8, replacing NaN with 'UNKNOWN'.
        cols: location, page_count
        """
        try:
            # Filter rows where page_count > 8
            filtered_df = df[df['page_count'] > 8]

            # Extract unique location values and return as a list
            return filtered_df['location'].unique().tolist()
        
        except Exception as e:
            return []


    def pp13_exe(df:pd.DataFrame):
        filtered_df = df[df['currency'].str.lower() == 'dollars']
        res = filtered_df['sponsor'].dropna().unique().tolist()
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
    
    def pp19_exe(df):
        """
        Determine the average number of pages for menus across different venues.
        cols: page_count, venue
        """
        try:
            # Convert page_count to numeric, coercing errors to NaN
            df['page_count'] = pd.to_numeric(df['page_count'], errors='coerce')

            # Drop rows with missing values after conversion
            df_clean = df.dropna(subset=['page_count', 'venue'])

            # Group by venue and calculate mean page count
            venue_avg_pages = df_clean.groupby('venue')['page_count'].mean()

            # Convert to dictionary format with venue as key and average pages as value
            return venue_avg_pages.to_dict()

        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def pp20_exe(df):
        """
        Return a list of unique sponsors.
        cols: sponsor
        """
        try:
            # Extract unique sponsor values and return as a list
            return df['sponsor'].dropna().unique().tolist()
        
        except Exception as e:
            return []

    
    def pp21_exe(df):
        """
        Count the number of rows where the event is marked as banquet.
        cols: event
        """
        # Convert event column to lowercase for case-insensitive comparison
        banquet_count = df[df['event'].str.lower() == 'banquet'].shape[0]
        return banquet_count
    
    def pp22_exe(df):
        """
        Count the number of unique occasions.
        cols: occasion
        """
        unique_occasions_count = df['occasion'].nunique()
        return unique_occasions_count
    
    def pp23_exe(df):
        """
        Find the top three venues with the highest average dish count.
        cols: venue, dish_count
        """
        try:
            # Convert page_count to numeric, coercing errors to NaN
            df['dish_count'] = pd.to_numeric(df['dish_count'], errors='coerce')

            # Drop rows with missing values after conversion
            df_clean = df.dropna(subset=['dish_count', 'venue'])
            # Calculate average dish count per venue
            venue_avg_dishes = df_clean.groupby('venue')['dish_count'].mean()
            # Sort venues by average dish count in descending order and get top 3
            top_three_venues = venue_avg_dishes.nlargest(3).index.tolist()
            return top_three_venues
        except:
            return []
    
    def pp24_exe(df):
        """
        Return a list of unique status values, excluding NaN values.
        cols: status
        """
        try:
            # Extract unique status values, drop NaN, and return as a list
            return df['status'].dropna().unique().tolist()
        
        except Exception as e:
            return []

    def pp25_exe(df):
        """
        Return a list of unique currency values in lowercase, excluding NaN values.
        cols: currency
        """
        try:
            # Convert currency column to lowercase and drop NaN values
            df['currency'] = df['currency'].str.lower()

            # Extract unique currency values and return as a list
            return df['currency'].dropna().unique().tolist()
        
        except Exception as e:
            return []

    def pp26_exe(df):
        """
        Count the number of menus published per year.
        cols: date
        """
        try:
            # Convert date column to datetime, coercing errors to NaT (invalid values become NaN)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Drop rows where date conversion failed (NaT values)
            df_clean = df.dropna(subset=['date'])

            # Extract year and count occurrences
            yearly_counts = df_clean['date'].dt.year.value_counts().sort_index()

            # Convert result to dictionary
            return yearly_counts.to_dict()

        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def pp27_exe(df):
        """
        Evaluate whether menus created in later years (e.g., after 1950) tend to have higher dish count-to-page count ratios.
        cols: date, dish_count, page_count
        """
        try:
            # Convert page_count to numeric, coercing errors to NaN
            df['dish_count'] = pd.to_numeric(df['dish_count'], errors='coerce')
            df['page_count'] = pd.to_numeric(df['page_count'], errors='coerce')
            # Convert date column to datetime, coercing errors to NaT
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df_clean = df.dropna(subset=['date', 'dish_count', 'page_count'])

            # Extract year from date
            df_clean['year'] = df_clean['date'].dt.year

            # Count rows where year < 1950
            pre_1950_count = (df_clean['year'] < 1950).sum()

            return int(pre_1950_count)  # Ensure it's an int for JSON serialization

        except Exception as e:
            print(f"Error: {e}")
            return None

    
    def pp28_exe(df):
        """
        Identify venues where the page count exceeds 10.
        cols: venue, page_count
        """
        # Filter venues where page count > 10
        try:
            # Convert page_count to numeric, coercing errors to NaN
            df['page_count'] = pd.to_numeric(df['page_count'], errors='coerce')

            # Filter out NaN values before comparison
            high_page_venues = df[df['page_count'] > 10]['venue'].dropna().unique().tolist()

            return high_page_venues

        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def pp29_exe(df):
        """
        Return a list of unique occasion values that contain 'daily' (case-insensitive).
        cols: occasion
        """
        try:
            # Convert occasion to string and lowercase for case-insensitive comparison
            df['occasion'] = df['occasion'].astype(str).str.lower()

            # Filter occasions that contain 'daily'
            daily_occasions = df[df['occasion'].str.contains('daily', na=False)]['occasion']

            # Extract unique occasion values and return as a list
            return daily_occasions.dropna().unique().tolist()
        
        except Exception as e:
            return []

    
    def pp30_exe(df):
        """
        Return a list of unique currency values in lowercase, excluding empty and NaN values.
        cols: currency
        """
        try:
            # Convert currency column to lowercase
            df['currency'] = df['currency'].str.lower()

            # Remove empty strings and NaN values
            df_cleaned = df['currency'].replace('', np.nan).dropna()

            # Extract unique currency values and return as a list
            return df_cleaned.unique().tolist()
        
        except Exception as e:
            return []

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
    
    def pp35_exe(df):
        """
        Count the unique types of inspections.
        cols: inspection type
        """
        # Get count of unique inspection types
        unique_inspection_count = df['Inspection Type'].astype(str).str.strip().str.lower().nunique()
        return unique_inspection_count

    def pp36_exe(df):
        try:
            failed_inspections_7eleven = df[(df['DBA Name'].str.lower() == '7-eleven'.lower()) & (df['Results'].str.lower() == 'fail')]['Inspection ID'].tolist()
            return failed_inspections_7eleven
        except:
            return None

    def pp37_exe(df):
        try:
            df['Results'] = df['Results'].astype(str).str.lower()
            # Group by DBA_Name and calculate the passing rate
            passing_rate = (
                df.groupby('DBA Name')
                .apply(lambda x: (x['Results'] == 'pass').sum() / len(x))
                .reset_index(name='Passing_Rate')
            )
            best_brand_name = passing_rate.sort_values(by='Passing_Rate', ascending=False).iloc[0]['DBA Name']
            return best_brand_name
        except:
            return None


    def pp38_exe(df):
        df['Risk'] = df['Risk'].astype(str).str.lower()
        unique_low_risk_facility_types = df[df['Risk'] == 'risk 3 (low)']['Facility Type'].unique()
        res = unique_low_risk_facility_types.tolist()
        return res 


    def pp39_exe(df):
        unique_high_risk_facility_types = df[df['Risk'].str.lower() == 'risk 1 (high)']['Facility Type'].unique()
        res = unique_high_risk_facility_types.tolist()
        return res 

    def pp40_exe(df):
        try:
            most_frequent_risk_by_type = df.groupby('Facility Type')['Risk'].agg(lambda x: x.value_counts().idxmax()).reset_index().to_json()
            return most_frequent_risk_by_type
        except:
            return None

    def pp41_exe(df):
        df['Risk'] = df['Risk'].astype(str)
        high_risk_facilities = df[df['Risk'].str.contains('risk 1', case=False, na=False)]
        unique_facility_types = high_risk_facilities['Facility Type'].dropna().unique().tolist()
        return [ftype for ftype in unique_facility_types if ftype.strip().upper() != "UNKNOWN"]

    def pp42_exe(df):
        try:
            high_risk_groceries_count = df[(df['Facility Type'].str.lower() == 'grocery store') & 
                                    (df['Risk'].str.contains('risk 1', case=False))].shape[0]
        except:
            high_risk_groceries_count = df[(df['Facility Type'].astype(str).str.lower() == 'grocery store') & 
                                (df['Risk'].astype(str).str.contains('risk 1', case=False))].shape[0]
        return high_risk_groceries_count
    
    def pp43_exe(df):
        """
        Find the most common result (pass/fail) for each facility type
        cols: Facility Type, Results
        """
        # Group by Facility Type and get most common Result
        most_common_results = df.groupby('Facility Type')['Results'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        return most_common_results.to_json()
    
    def pp44_exe(df):
        """
        Calculate the proportion of inspections that result in a positive outcome (pass)
        cols: Results
        """
        # Convert Results to lowercase for consistent comparison
        df['Results'] = df['Results'].astype(str).str.lower()
        
        # Calculate total number of inspections
        total_inspections = len(df)
        
        # Count number of passing inspections
        passing_inspections = (df['Results'] == 'pass').sum()
        
        # Calculate proportion
        pass_proportion = passing_inspections / total_inspections
        
        return pass_proportion
    
    def pp45_exe(df):
        """
        Calculate the average number of inspections per facility type
        cols: Facility Type, Inspection ID
        """
        # Group by Facility Type and count unique Inspection IDs
        inspections_per_type = df.groupby('Facility Type')['Inspection ID'].count().reset_index()
        
        # Calculate the average number of inspections across all facility types
        average_inspections = inspections_per_type['Inspection ID'].mean()
        
        return average_inspections
    
    def pp46_exe(df):
        """
        Calculate correlation between risk level and risk label.
        cols: Risk
        """
        # Convert Risk column to string type and standardize case
        df['Risk'] = df['Risk'].astype(str).str.lower()
        # Get unique risk values and count occurrences
        risk_counts = df['Risk'].value_counts().to_dict()
        return risk_counts
    
    def pp47_exe(df):
        """
        Find facility type with highest failure rate
        cols: Facility Type, Results
        """
        # Convert Results to lowercase for consistent comparison
        df['Results'] = df['Results'].astype(str).str.lower()
        
        # Group by Facility Type and calculate failure rate
        failure_rates = df.groupby('Facility Type').agg(
            total_inspections=('Results', 'count'),
            failed_inspections=('Results', lambda x: (x == 'fail').sum())
        )
        
        failure_rates['failure_rate'] = failure_rates['failed_inspections'] / failure_rates['total_inspections']
        
        # Get facility type with highest failure rate
        highest_failure_type = failure_rates['failure_rate'].idxmax()
        
        return highest_failure_type
    
    def pp48_exe(df):
        """
        Find the franchise (DBA Name) with the most frequent inspections
        cols: DBA Name, Inspection ID
        """
        # Group by DBA Name and count inspections
        inspection_counts = df.groupby('DBA Name')['Inspection ID'].count()
        
        # Get the franchise with maximum inspections
        most_inspected = inspection_counts.idxmax()
        
        return most_inspected

    def pp49_exe(df):
        try:
            safest_school_restaurants_count = df[(df['Facility Type'].str.lower() == 'school') & 
                                        (df['Risk'].str.contains('risk 1', case=False)) & 
                                        (df['Results'].str.lower() == 'pass')].shape[0]
        except:
            safest_school_restaurants_count = df[(df['Facility Type'].astype(str).str.lower() == 'school') & 
                                        (df['Risk'].astype(str).str.contains('risk 1', case=False)) & 
                                        (df['Results'].astype(str).str.lower() == 'pass')].shape[0]
        return safest_school_restaurants_count
    
    def pp50_exe(df):
        """
        Count the number of inspections conducted as part of a complaint.
        cols: Inspection Type
        """
        # Convert to lowercase for case-insensitive comparison and count complaint inspections
        complaint_count = df[df['Inspection Type'].str.lower().str.contains('complaint', na=False)].shape[0]
        return complaint_count
    
    def pp51_exe(df):
        """
        Find facilities with positive inspection results.
        cols: Facility Type, Results
        """
        # Convert Results to lowercase for consistent comparison
        df['Results'] = df['Results'].astype(str).str.lower()
        
        # Filter for results containing 'pass' and get unique facility types
        passing_facilities = df[df['Results'].str.contains('pass', na=False)]['Facility Type'].dropna().unique().tolist()
        
        return passing_facilities

    def pp52_exe(df):
        """
        Return a list of addresses for facilities that have 'Risk 3' and passed inspection.
        """
        # Ensure 'Risk' and 'Results' are strings and perform case-insensitive filtering
        safe_facilities = df[
            df['Risk'].str.contains('risk 3', case=False, na=False) & 
            df['Results'].str.lower().eq('pass')
        ]

        # Extract addresses
        safe_addresses = safe_facilities['Address'].dropna().tolist()
        
        return safe_addresses
    
    def pp53_exe(df):
        """
        List unique inspection results for all businesses with Risk level 1.
        Columns: Risk, Results
        """
        # Convert Risk to lowercase for case-insensitive comparison
        df['Risk'] = df['Risk'].astype(str).str.lower()
        
        # Filter rows where 'Risk' contains 'risk 1'
        risk1_results = df[df['Risk'].str.contains('risk 1', case=False, na=False)]['Results'].dropna().unique().tolist()
        
        return risk1_results
    
    def pp54_exe(df):
        """
        Determine the most recent year of inspection dates.
        cols: inspection date
        """
        # Convert inspection date to datetime if not already
        try:
            # Convert cleaned date column to datetime, coercing errors to NaT
            df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], errors='coerce')

            # Extract the most recent year, ignoring NaT values
            most_recent_year = df['Inspection Date'].dt.year.max()

            return int(most_recent_year) if not pd.isna(most_recent_year) else None

        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def pp55_exe(df):
        """
        Identify businesses with the most frequent inspections.
        cols: DBA Name
        """
        # Count inspections per business
        inspection_counts = df['DBA Name'].value_counts()
        
        # Get businesses with the maximum number of inspections
        max_inspections = inspection_counts.max()
        most_inspected = inspection_counts[inspection_counts == max_inspections].index.tolist()
        
        return most_inspected
    def pp56_exe(df):
        """
        List license IDs with the best inspection results (Pass).
        cols: License #, Results
        """
        # Convert Results to lowercase for case-insensitive comparison
        df['Results'] = df['Results'].astype(str).str.lower()
        
        # Filter for passing results and get unique license numbers
        passing_licenses = df[df['Results'] == 'pass']['License #'].unique().tolist()
        
        return passing_licenses
    
    def pp57_exe(df):
        """
        Identify businesses where the inspection risk level is Risk 3.
        cols: DBA Name, Risk, Inspection Date
        """
        # Convert Risk to lowercase for case-insensitive comparison
        df['Risk'] = df['Risk'].astype(str).str.lower()
        
        # Filter businesses with Risk 3 (low)
        risk3_businesses = df[df['Risk'].str.contains('risk 3')]['DBA Name'].unique().tolist()
        
        return risk3_businesses
    
    def pp58_exe(df):
        """
        Identify the most common risk levels for inspections conducted in restaurant.
        cols: Facility Type, Risk
        """
        # Filter for restaurant facilities
        restaurant_df = df[df['Facility Type'].str.lower() == 'restaurant']
        
        # Get the most common risk level(s)
        risk_counts = restaurant_df['Risk'].value_counts()
        max_count = risk_counts.max()
        most_common_risks = risk_counts[risk_counts == max_count].index.tolist()
        
        return most_common_risks
    
    def pp59_exe(df):
        """
        Identify businesses inspected on the most recent date.
        cols: DBA Name, Inspection Date
        """
        try:
            # Get the most recent inspection date
            df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], errors='coerce')
            most_recent_date = df['Inspection Date'].max()
            if pd.isna(most_recent_date):
                return None
            
            # Filter businesses inspected on the most recent date
            recent_businesses = df[df['Inspection Date'] == most_recent_date]['DBA Name'].unique().tolist()
            
            return recent_businesses
        except:
            return None
    
    def pp60_exe(df):
        """
        Determine the facility types that are inspected on the most recent date.
        cols: Facility Type, Inspection Date
        """
        try:
            # Convert Inspection Date to datetime
            df['Inspection Date'] = pd.to_datetime(df['Inspection Date'], errors='coerce')

            # Get the most recent inspection date
            most_recent_date = df['Inspection Date'].max()

            if pd.isna(most_recent_date):
                return None  # No valid date found

            # Find facility types inspected on the most recent date
            recent_types = df[df['Inspection Date'] == most_recent_date]['Facility Type'].dropna().unique().tolist()

            return recent_types if recent_types else None  # Return None if the list is empty

        except Exception as e:
            print(f"Error: {e}")
            return None
    
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
        honolulu_loans_count = df[df['City'].astype(str).str.lower() == 'honolulu'].shape[0]
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
        df = df[df['Gender'].str.lower() != 'unanswered']
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
        return str(highest_zip_codes.iloc[0]['Zip']) if not highest_zip_codes.empty else None
    
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
        try:
            df = df[df['RaceEthnicity'].str.lower() != "unanswered"]
            race_loan_totals = df.groupby('RaceEthnicity').agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()

            # Sort by Total Loan Amount in descending order
            highest_race = race_loan_totals.sort_values(by='TotalLoanAmount', ascending=False)
            return highest_race.iloc[0]['RaceEthnicity']
        except:
            return None
    
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
        geo_distribution = df.groupby(['City', 'Zip']).agg(TotalLoanAmount=('LoanAmount', 'sum')).reset_index()
        geo_distribution = geo_distribution.loc[geo_distribution['TotalLoanAmount'].idxmax()]
        # geo_distribution = geo_distribution.sort_values(by='TotalLoanAmount', ascending=False)
        return [str(x) for x in geo_distribution[['City', 'Zip']].tolist()]
    
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
        try:
            df['last_appeared'] = pd.to_datetime(df['last_appeared'], errors='coerce')
            df['first_appeared'] = pd.to_datetime(df['first_appeared'], errors='coerce')

            # Calculate the difference in years
            df = df.dropna(subset=['first_appeared', 'last_appeared'])
            df['duration'] = (df['last_appeared'] - df['first_appeared']).dt.days

            # Filter to identify dishes with the shortest duration
            shortest_duration = df['duration'].min()
            shortest_duration_dishes = df[df['duration'] == shortest_duration]
            return shortest_duration_dishes['name'].tolist() 
        except:
            return []
        
    
    def pp94_exe(df):
        """
        Identify which dishes have been on the menu for the longest duration, based on their 'first_appeared' and 'last_appeared' dates.
        cols:first_appeared, last_appeared
        """
        try:
            df['last_appeared'] = pd.to_datetime(df['last_appeared'], errors='coerce')
            df['first_appeared'] = pd.to_datetime(df['first_appeared'], errors='coerce')
            df = df.dropna(subset=['first_appeared', 'last_appeared'])

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
        try:
            df['first_appeared'] = pd.to_datetime(df['first_appeared'], errors='coerce')
            df = df.dropna(subset=['first_appeared', 'last_appeared'])
            dishes_2000 = df[df['first_appeared'].dt.year < 2000]
            # Find the earliest first_appeared date
            min_date = dishes_2000['first_appeared'].min()

            # Get all dish names that first appeared on that earliest date
            earliest_dishes = dishes_2000[dishes_2000['first_appeared'] == min_date]['name'].tolist()

            return earliest_dishes
        except Exception as e:
            return str(e)

    def pp101_exe(df):
        """
        Identify which dishes were the first to appear on the menu.
        cols: name, first_appeared
        """
        try:
            # Convert first_appeared to datetime format
            df['first_appeared'] = pd.to_datetime(df['first_appeared'], errors='coerce')

            # Drop rows with NaN in first_appeared
            df = df.dropna(subset=['first_appeared'])

            # Find the earliest appearance date
            earliest_date = df['first_appeared'].min()

            # Get all dish names that appeared on that earliest date
            first_dishes = df[df['first_appeared'] == earliest_date]['name'].tolist()

            return first_dishes
        except Exception as e:
            return str(e)  # Return error message for debugging

    
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
        df['highest_price'] = pd.to_numeric(df['highest_price'], errors='coerce')
        df['lowest_price'] = pd.to_numeric(df['lowest_price'], errors='coerce')
        df['price_difference'] = df['highest_price'] - df['lowest_price']

        df_valid = df.dropna(subset=['price_difference'])

        if df_valid.empty:
            return None  # If all rows were NaN, return None

        # Identify the dish with the lowest price difference
        lowest_price_difference_dish = df_valid.loc[df_valid['price_difference'].idxmin()]

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
        df = df.dropna(subset=['name', 'highest_price', 'lowest_price'])

        # Ensure prices are numeric
        df[['highest_price', 'lowest_price']] = df[['highest_price', 'lowest_price']].apply(pd.to_numeric, errors='coerce')

        # Check if df is still valid after dropping missing values
        if df.empty:
            return None
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
    
    
    def pp111_exe(df):
        """
        Calculate the average delay in departure times across all flights.
        cols: sched_dep_time, act_dep_time
        """
        try:
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)
            df['delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60

            delayed_flights = df[df['delay'] > 0]

            # Extract unique, non-null scheduled departure times
            unique_times = delayed_flights['sched_dep_time'].dropna().unique().tolist()

            return unique_times
        except:
            return None

    def pp112_exe(df):
        """
        Calculate the average flight duration based on scheduled departure and arrival times.
        cols: sched_dep_time, sched_arr_time
        """
        try:
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_sched_arr_time'] = df['sched_arr_time'].apply(safe_parse_datetime)

            df['duration'] = (df['parsed_sched_arr_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
            return df['duration'].mean(skipna=True)
        except:
            return None


    def pp113_exe(df):
        """
        Determine the airline carrier with the best on-time departure performance.
        cols: src, sched_dep_time, act_dep_time
        """
        try:
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

            df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
            avg_delay_per_airline = df.groupby('src')['departure_delay'].mean()

            return avg_delay_per_airline.idxmin() if not avg_delay_per_airline.empty else None
        except:
            return None


    def pp114_exe(df):
        """
        Count the number of unique airline carriers.
        cols: src
        """
        try:
            # Drop NaN values and extract unique airline carriers
            unique_carriers = df['src'].dropna().unique().tolist()
            return unique_carriers
        except Exception as e:
            return str(e)  # Return error message for debugging


    def pp115_exe(df):
        """
        Identify the most common flight number.
        cols: flight
        """
        try:
            # Drop NaN values and extract unique airline carriers
            unique_flights = df['flight'].dropna().unique().tolist()
            return unique_flights
        except Exception as e:
            return str(e)  # Return error message for debugging


    # def pp116_exe(df):
    #     """
    #     Count flights scheduled to arrive after 6:00 PM.
    #     cols: sched_arr_time
    #     """
    #     try:
    #         count_after_6pm = df[df['sched_arr_time'].str.contains("T18:|T19:|T20:|T21:|T22:|T23:", regex=True, na=False)].shape[0]
    #         return int(count_after_6pm)
    #     except:
    #         return None
    def pp116_exe(df):
        """
        Return a list of sched_arr_time for flights scheduled to arrive after 6:00 PM.
        cols: sched_arr_time
        """
        try:
            # Ensure sched_arr_time is not NaN before applying regex
            df_cleaned = df.dropna(subset=['sched_arr_time']).copy()

            flights_after_6pm = df_cleaned[df_cleaned['sched_arr_time'].str.contains(
                r"T18:|T19:|T20:|T21:|T22:|T23:", regex=True, na=False
            )]

            return flights_after_6pm['sched_arr_time'].dropna().tolist()
        
        except Exception as e:
            return []



    def pp117_exe(df):
        """
        Count on-time arrivals per airline carrier.
        cols: src, sched_arr_time, act_arr_time
        """
        try:
            df['on_time'] = df['sched_arr_time'] == df['act_arr_time']
            # Filter only carriers with at least one on-time arrival
            on_time_carriers = df[df['on_time']]['src'].dropna().unique().tolist()

            return on_time_carriers
        except Exception as e:
            return str(e)

    def pp118_exe(df):
        """
        Count how many flights are "delayed."
        cols: sched_dep_time, act_dep_time
        """
        try:
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

            # Calculate delay only for valid timestamps
            df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60

            # Filter delayed flights (departure_delay > 0)
            delayed_flights = df[df['departure_delay'] > 0]

            # Extract unique hours from scheduled departure time
            delayed_hours = sorted(delayed_flights['sched_dep_time'].str[11:13].dropna().unique().tolist())

            return delayed_hours
        except:
            return []


    # def pp119_exe(df):
    #     """
    #     Count flights that arrived in the AM.
    #     cols: sched_arr_time
    #     """
    #     try:
    #         am_count = df[df['sched_arr_time'].str.contains("T0[0-9]:|T10:|T11:", regex=True, na=False)].shape[0]
    #         return int(am_count)
    #     except:
    #         return None
    def pp119_exe(df):
        """
        Return a list of sched_arr_time for flights scheduled to arrive in the AM.
        cols: sched_arr_time
        """
        try:
            # Ensure sched_arr_time is not NaN before applying regex
            df_cleaned = df.dropna(subset=['sched_arr_time']).copy()

            flights_in_am = df_cleaned[df_cleaned['sched_arr_time'].str.contains(
                r"T0[0-9]:|T10:|T11:", regex=True, na=False
            )]

            return flights_in_am['sched_arr_time'].dropna().tolist()
        
        except Exception as e:
            return []


    # def pp120_exe(df):
    #     """
    #     Identify the airline carrier with the most delayed flights.
    #     cols: src, sched_dep_time, act_dep_time
    #     """
    #     try:
    #         df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
    #         df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

    #         df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
    #         delayed_flights = df[df['departure_delay'] > 0]

    #         return delayed_flights['src'].value_counts().idxmax() if not delayed_flights.empty else None
    #     except:
    #         return None
    def pp120_exe(df):
        """
        Return a list of airline carriers (src) that have delayed flights.
        cols: src, sched_dep_time, act_dep_time
        """
        try:
            # Ensure datetime parsing handles NaN values safely
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)

            # Drop rows where either sched_dep_time or act_dep_time is missing
            df_cleaned = df.dropna(subset=['parsed_sched_dep_time', 'parsed_act_dep_time']).copy()

            # Compute departure delay
            df_cleaned['departure_delay'] = (df_cleaned['parsed_act_dep_time'] - df_cleaned['parsed_sched_dep_time']).dt.total_seconds() / 60

            # Filter flights that were delayed (departure_delay > 0)
            delayed_flights = df_cleaned[df_cleaned['departure_delay'] > 0]

            # Return a list of unique airline carriers with delayed flights
            return delayed_flights['src'].dropna().unique().tolist()
        
        except Exception as e:
            return []

    # def pp121_exe(df):
    #     """
    #     Count flights where the actual departure time is earlier than scheduled.
    #     cols: sched_dep_time, act_dep_time
    #     """
    #     try:
    #         df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
    #         df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

    #         df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
    #         return int((df['departure_delay'] < 0).sum(skipna=True))
    #     except:
    #         return None
     
    def pp121_exe(df):
        """
        Return a list of actual departure times (act_dep_time) that are earlier than their scheduled departure times (sched_dep_time).
        cols: sched_dep_time, act_dep_time
        """
        try:
            # Ensure that datetime parsing handles NaN values safely
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)

            # Drop rows where either time is NaT (i.e., missing) before calculation
            df_cleaned = df.dropna(subset=['parsed_sched_dep_time', 'parsed_act_dep_time']).copy()

            # Compute departure delay only on clean data
            df_cleaned['departure_delay'] = (df_cleaned['parsed_act_dep_time'] - df_cleaned['parsed_sched_dep_time']).dt.total_seconds() / 60

            # Filter flights where actual departure time is earlier than scheduled
            early_departures = df_cleaned[df_cleaned['departure_delay'] < 0]

            # Return a list of actual departure times
            return early_departures['act_dep_time'].dropna().tolist()
        
        except Exception as e:
            return []

    def pp122_exe(df):
        """
        Return a dictionary {act_arr_time: act_dep_time} for flights where actual arrival time is before actual departure time.
        cols: act_dep_time, act_arr_time
        """
        try:
            # Ensure datetime parsing handles NaN values safely
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)
            df['parsed_act_arr_time'] = df['act_arr_time'].apply(lambda x: safe_parse_datetime(x) if pd.notna(x) else pd.NaT)

            # Drop rows where either actual departure or arrival time is missing
            df_cleaned = df.dropna(subset=['parsed_act_dep_time', 'parsed_act_arr_time']).copy()

            # Filter flights where actual arrival time is before actual departure time
            invalid_flights = df_cleaned[df_cleaned['parsed_act_arr_time'] < df_cleaned['parsed_act_dep_time']]

            # Return a dictionary {act_arr_time: act_dep_time}
            return dict(zip(invalid_flights['act_arr_time'], invalid_flights['act_dep_time']))
        
        except Exception as e:
            return {}

    def pp123_exe(df):
        """
        Analyze the trend of delayed departures by hour of the day.
        cols: sched_dep_time, act_dep_time
        """
        try:
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

            # df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
            # df['hour_of_day'] = df['parsed_sched_dep_time'].dt.hour  # Extract hour from scheduled departure time

            df = df.dropna(subset=['parsed_sched_dep_time', 'parsed_act_dep_time'])

            df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
            return df.groupby(df['parsed_sched_dep_time'].dt.hour)['departure_delay'].mean().to_dict()

        except Exception as e:
            print(f"Error: {e}")
            return "Error in calculation"
    
    def pp124_exe(df):
        """
        Return a sorted list of unique hours where flights arrived on time.
        cols: sched_arr_time, act_arr_time
        """
        try:
            # Identify on-time arrivals
            df['on_time'] = df['sched_arr_time'] == df['act_arr_time']

            # Extract unique hours from on-time scheduled arrival times
            on_time_hours = sorted(df[df['on_time']]['sched_arr_time'].str[11:13].dropna().unique().tolist())

            return on_time_hours
        
        except Exception as e:
            return []


    # def pp124_exe(df):
    #     """
    #     Identify the hour with the highest number of on-time arrivals.
    #     cols: sched_arr_time, act_arr_time
    #     """
    #     try:
    #         df['on_time'] = df['sched_arr_time'] == df['act_arr_time']
    #         best_hour_for_arrivals = df[df['on_time']]['sched_arr_time'].str[11:13].value_counts().idxmax()
    #         return int(best_hour_for_arrivals)
    #     except:
    #         return None


    # def pp125_exe(df):
    #     """
    #     Identify the airline carrier with the highest average departure delay time.
    #     cols: src, sched_dep_time, act_dep_time
    #     """
    #     try:
    #         df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
    #         df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

    #         df['departure_delay'] = (df['parsed_act_dep_time'] - df['parsed_sched_dep_time']).dt.total_seconds() / 60
    #         avg_delay_by_src = df.groupby('src')['departure_delay'].mean()

    #         if avg_delay_by_src.empty:
    #             return None  # No valid data available

    #         return str(avg_delay_by_src.idxmax())  # Ensure string output for JSON serialization
    #     except:
    #         return None
     
    def pp125_exe(df):
        """
        Return a list of unique airline carriers (src) where both scheduled and actual departure times are correctly parsed.
        cols: src, sched_dep_time, act_dep_time
        """
        try:
            # Parse times only if they match ISO 8601 format
            df['parsed_sched_dep_time'] = df['sched_dep_time'].apply(safe_parse_datetime)
            df['parsed_act_dep_time'] = df['act_dep_time'].apply(safe_parse_datetime)

            # Keep only rows where parsing was successful (i.e., not original values)
            valid_rows = df[(pd.to_datetime(df['parsed_sched_dep_time'], errors='coerce').notna()) &
                            (pd.to_datetime(df['parsed_act_dep_time'], errors='coerce').notna())]

            # Return a list of unique src values
            return valid_rows['src'].dropna().unique().tolist()
        
        except Exception as e:
            return []



    def pp126_exe(df):
        """
        Return a list of unique arrival times where scheduled arrival time equals actual arrival time.
        cols: sched_arr_time, act_arr_time
        """
        try:
            # Drop rows where either sched_arr_time or act_arr_time is missing
            df_cleaned = df.dropna(subset=['sched_arr_time', 'act_arr_time']).copy()

            # Filter flights where scheduled and actual arrival times differ
            same_arrival_times = df_cleaned[df_cleaned['sched_arr_time'] == df_cleaned['act_arr_time']]

            # Return a dictionary {sched_arr_time: act_arr_time}
            return same_arrival_times['sched_arr_time'].dropna().unique().tolist()
                    
        except Exception as e:
            return {}


    def pp127_exe(df):
        """
        Count the number of hospitals per city.
        """
        hospital_count_per_city = df['City'].value_counts()
        return len(hospital_count_per_city)


    def pp128_exe(df):
        """
        Explore whether larger cities host more hospitals.
        """
        city_names = df['City'].unique().tolist()
        return city_names


    def pp129_exe(df):
        """
        Calculate the average number of hospitals per county.
        """
        county_names = df['CountyName'].unique().tolist()
        return county_names


    def pp130_exe(df):
        """
        Determine the top 3 counties whose hospital type is "acute care hospitals" offering emergency services.
        """
        df['HospitalType'] = df['HospitalType'].astype(str).str.lower()
        df['EmergencyService'] = df['EmergencyService'].astype(str).str.lower()
        top_3_counties = df[(df['HospitalType'] == 'acute care hospitals') & (df['EmergencyService'] == 'yes')]['CountyName'].value_counts().head(3)
        return top_3_counties.to_dict()


    def pp131_exe(df):
        """
        Identify the hospital ownership types that are most common in cities with multiple ZIP codes.
        """
        common_ownership_in_cities = df.groupby('City')['HospitalOwner'].nunique().sort_values(ascending=False)
        if common_ownership_in_cities.empty:
            return None
        else:
            return common_ownership_in_cities.to_dict()


    def pp132_exe(df):
        """
        Compute the average length of hospital names grouped by hospital type and ownership.
        """
        # avg_name_length = df.groupby(['HospitalType', 'HospitalOwner'])['HospitalName'].apply(lambda x: x.str.len().mean())
        # return avg_name_length
        try:
            # Count unique values in the HospitalType column
            df['HospitalType'] = df['HospitalType'].str.lower()
            num_unique_types = df['HospitalType'].nunique()

            # Ensure output is a standard Pyxthon int for JSON serialization
            return int(num_unique_types)

        except Exception as e:
            print(f"Error: {e}")
            return None


    def pp133_exe(df):
        """
        Count the number of hospitals offering emergency services by "acute care hospitals" hospital type.
        """
        try:
            df['HospitalType'] = df['HospitalType'].str.lower()
            df['EmergencyService'] = df['EmergencyService'].str.lower()
            emergency_services_count = df[(df['HospitalType'] == 'acute care hospitals') & (df['EmergencyService'] == 'yes')].shape[0]
            return emergency_services_count
        except:
            return None

    def pp134_exe(df):
        """
        Determine the most frequent combinations of hospital type and ownership.
        """
        frequent_combinations = df.groupby(['HospitalType', 'HospitalOwner']).size().idxmax()
        return frequent_combinations


    def pp135_exe(df):
        """
        Count the number of hospitals by county, grouped by the presence or absence of emergency services.
        """
        hospitals_by_county = df.groupby(['CountyName', 'EmergencyService']).size()
        return len(hospitals_by_county.to_dict())


    def pp136_exe(df):
        """
        Find the number of cities with hospitals owned by voluntary non-profits and offering emergency services.
        """
        try:
            df['EmergencyService'] = df['EmergencyService'].str.lower()
            df['HospitalOwner'] = df['HospitalOwner'].str.lower()
            cities_with_voluntary_nonprofits = df[(df['HospitalOwner'] == 'voluntary non-profit - private') & (df['EmergencyService'] == 'yes')]['City'].nunique()
            return cities_with_voluntary_nonprofits
        except:
            return None

    def pp137_exe(df):
        """
        Find hospital types among different counties.
        """
        hospital_types_by_county = df.groupby('CountyName')['HospitalType'].apply(lambda x: x.unique().tolist())
        return len(hospital_types_by_county.to_dict())


    def pp138_exe(df):
        """
        Identify cities where critical access hospitals are primarily government-owned.
        """
        df['HospitalOwner'] = df['HospitalOwner'].str.lower()
        # Filter hospitals that are 'Critical Access Hospitals' and government-owned
        # Normalize HospitalType to avoid any leading/trailing space issues
        df['HospitalType'] = df['HospitalType'].str.strip().str.upper()  # Convert to uppercase for case-insensitive match

        # Filter hospitals that are 'ACUTE CARE HOSPITALS' and government-owned
        government_owned_acute_care = df[
            (df['HospitalType'] == 'ACUTE CARE HOSPITALS') &  # Ensure correct hospital type
            (df['HospitalOwner'].str.contains(r'\bgovernment\b', regex=True))  # Match 'government' as a whole word
        ]['City'].unique()

        return government_owned_acute_care.tolist()


    def pp139_exe(df):
        """
        Count ZIP codes based on the diversity of hospital types and ownerships.
        """
        zip_code_diversity = df.groupby('ZipCode')[['HospitalType', 'HospitalOwner']].nunique().sum(axis=1)
        return zip_code_diversity.to_dict()

    def pp140_exe(df):
        """
        Determine the top 3 cities with the highest ratio of acute care hospitals to total hospitals.
        """
        df['HospitalType'] = df['HospitalType'].str.lower()
        top_3_cities = df[df['HospitalType'] == 'acute care hospitals']['City'].value_counts(normalize=True).head(3)
        return top_3_cities.to_dict()

    def pp141_exe(df):
        """
        Analyze how hospital types evolve across the ZIP code of one city.
        """
        hospital_type_evolution = df.groupby('City')['HospitalType'].apply(lambda x: x.unique().tolist()).to_dict()
        return hospital_type_evolution

    def pp142_exe(df):
        """
        Analyze trends in emergency service availability across hospital types with the highest hospital counts.
        """
        try:
            df['EmergencyService'] = df['EmergencyService'].astype(str).str.lower()
            hospital_type_with_most_emergency_service = df[df['EmergencyService'] == 'yes'].groupby('HospitalType').size().idxmax()
            return hospital_type_with_most_emergency_service
        except:
            return None

    def pp143_exe(df):
        """
        Analyze whether the hospital ownership changed among counties for the same type.
        """
        try:
            # Count unique values in the HospitalType column
            df['HospitalType'] = df['HospitalType'].str.lower()
            num_unique_types = df['HospitalType'].nunique()

            # Ensure output is a standard Pyxthon int for JSON serialization
            return int(num_unique_types)

        except Exception as e:
            print(f"Error: {e}")
            return None

    def pp144_exe(df):
        """
        Assess the relationship between hospital ownership and the likelihood of offering emergency services.
        """
        try:
            df['EmergencyService'] = df['EmergencyService'].str.lower()
            num_unique_types = df['EmergencyService'].nunique()

            # Ensure output is a standard Pyxthon int for JSON serialization
            return int(num_unique_types)

        except Exception as e:
            print(f"Error: {e}")
            return None

    def pp145_exe(df):
        """
        Investigate correlations between city size (estimated by the number of hospitals) and the prevalence of emergency services.
        """

        count_cities = df.groupby('City')['EmergencyService'].apply(lambda x: (x.str.lower() == 'yes').any()).sum() 
        return int(count_cities)
        

    def pp146_exe(df):
        """
        Explore the relationship between hospital types and their geographical distribution by ZIP code.
        """
        hospital_type_zip_code = df.groupby('HospitalType')['ZipCode'].nunique().sort_values(ascending=False)
        return hospital_type_zip_code.to_dict()

    def pp147_exe(df):
        """
        Analyze how ownership type correlates with hospital type.
        """
        try:
            # Count unique values in the HospitalType column
            df['HospitalType'] = df['HospitalType'].str.lower()
            num_unique_types = df['HospitalType'].nunique()

            # Ensure output is a standard Pyxthon int for JSON serialization
            return int(num_unique_types)

        except Exception as e:
            print(f"Error: {e}")
            return None

    def pp148_exe(df):
        """
        Retrieve the number of counties that offer the emergency services.
        """
        df['EmergencyService'] = df['EmergencyService'].str.lower()
        counties_with_emergency_services = df[df['EmergencyService'] == 'yes']['CountyName'].unique().tolist()
        return len(counties_with_emergency_services)

    def pp149_exe(df):
        """
        Filter for hospitals owned by governments with multiple hospital types.
        """
        try:
            # Convert HospitalOwner to lowercase for case-insensitive filtering
            df['HospitalOwner'] = df['HospitalOwner'].astype(str).str.lower()

            # Filter for government-owned hospitals
            government_hospitals = df[df['HospitalOwner'].str.contains('government', na=False)]

            # Count unique hospital types
            num_unique_types = government_hospitals['HospitalType'].nunique()

            return int(num_unique_types)  # Ensure JSON serialization compatibility

        except Exception as e:
            print(f"Error: {e}")
            return None

    def pp150_exe(df):
        """
        How many hospital types offer emergency services.
        """
        df['EmergencyService'] = df['EmergencyService'].str.lower()
        hospital_types_offering_emergency_services = df[df['EmergencyService'] == 'yes']['HospitalType'].unique().size
        return hospital_types_offering_emergency_services

    def pp151_exe(df):
        """Return the unique city names where the hospital owner is 'VOLUNTARY NON-PROFIT - PRIVATE'."""
        cities_with_voluntary_nonprofit_hospitals = df[
            df['HospitalOwner'].str.contains('voluntary non-profit - private', case=False, na=False)
        ]['City'].unique().tolist()
        return cities_with_voluntary_nonprofit_hospitals

    def pp152_exe(df):
        """
        Return a list of city names where government-owned hospitals dominate.
        cols: HospitalOwner, City
        """
        try:
            # Ensure HospitalOwner is in lowercase for standardization
            df['HospitalOwner'] = df['HospitalOwner'].str.lower()

            # Filter hospitals that are government-owned
            government_hospitals = df[df['HospitalOwner'].str.contains('government', na=False)]

            # Identify cities where government-owned hospitals exist
            dominant_cities = government_hospitals['City'].dropna().unique().tolist()

            return dominant_cities
        
        except Exception as e:
            return []

    
    # def pp152_exe(df):
    #     """
    #     Identify number of cities where government-owned hospitals dominate the hospital landscape.
    #     """
    #     df['HospitalOwner'] = df['HospitalOwner'].str.lower()
    #     government_dominant_cities = df[df['HospitalOwner'].str.contains('government')].groupby('City').size()
    #     return len(government_dominant_cities)

    # def pp153_exe(df):
    #     """
    #     Determine the number of hospitals offering emergency services.
    #     """
    #     df['EmergencyService'] = df['EmergencyService'].str.lower()
    #     emergency_service_count = (df['EmergencyService'] == 'yes').sum()
    #     return int(emergency_service_count)
    def pp153_exe(df):
        """
        Return a list of unique zip codes for hospitals offering emergency services.
        cols: EmergencyService, Zipcode
        """
        try:
            # Ensure EmergencyService column is in lowercase to standardize comparison
            df['EmergencyService'] = df['EmergencyService'].str.lower()

            # Filter hospitals that offer emergency services
            emergency_idx = df[df['EmergencyService'] == 'yes'].index

            # Return a list of unique zip codes
            return df.loc[emergency_idx, 'ZipCode'].dropna().unique().tolist()
        
        except Exception as e:
            return []


    # def pp154_exe(df):
    #     """
    #     return number of hospital owner.
    #     """
    #     hospital_owner_count = df['HospitalOwner'].nunique()
    #     return hospital_owner_count
    def pp154_exe(df):
        """
        Return a list of unique hospital owners.
        cols: HospitalOwner
        """
        try:
            # Extract unique hospital owners and return as a list
            return df['HospitalOwner'].dropna().unique().tolist()
        
        except Exception as e:
            return []


if __name__ == '__main__':
    qexecute = QExecute
    # Load queries contents
    query_contents = pd.read_csv('../purposes/all_purposes.csv')
    # model = 'dirty'
    # load results by LLMs
    model = "llama3.1"
    # model = "gemma2"
    # model = "mistral"
    # model = "gemma2base"
    llm_folder = f"Error_Analysis/{model}/datasets_llm"
    par_fp = "/projects/bces/lanl2/LLM4DC"
    pp_fp = par_fp+'/purposes/'+'llama_log.csv'
    # pp_fp = par_fp+'/purposes/'+'mistral_log.csv'
    pp_data = pd.read_csv(pp_fp)
    print(pp_data['ID'])
    assert len(pp_data)==20

    for _,row in pp_data.iterrows():
        query_id, query_content = row['ID'], row['Purposes']
        # row = query_contents[query_contents['ID'] == query_id]
        # if len(row) == 0:
        #     continue
        func = f'pp{query_id}_exe'
        print(func)
        print(f"current model: {model}")
        if query_id >126:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_hos_p{query_id}.csv'
        elif query_id >= 111 and query_id <=126:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_flights_p{query_id}.csv'
        elif query_id >= 92 and query_id <=110:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_dish_p{query_id}.csv'
        elif query_id >= 62 and query_id <= 91:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_ppp_p{query_id}.csv'
        elif query_id >= 31 and query_id <= 61:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_chi_p{query_id}.csv'
        elif query_id <31:
            target_path = f'{par_fp}/{llm_folder}/log_{model}_menu_p{query_id}.csv'
        print(target_path)
        target_df = pd.read_csv(target_path)
        
        answer = getattr(qexecute, func)(target_df)
        
        print('answer', type(answer), answer)
        result_single = {'pp_id': query_id,
                        'purpose': query_content,
                        'answer': answer}

        with open(f'log_answer_{model}.json', 'a') as f:
            f.write(json.dumps(result_single))
            f.write('\n')

