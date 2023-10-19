import json
import pandas as pd
import pandas.plotting as pdplt
import numpy as np
import polars as pl
import io
import boto3
import pandas as pd


def transform_data(in_B2, out_B1):
    WEIGHT_0STOPS = 1.5
    WEIGHT_1STOPS = 1/2
    
    df_b = in_B2
    df_b1 = out_B1
    df_b = df_b.join(df_b1, how='left', on=['RecordID'])
    OD_id = ['dest', 'orig', 'year','quarter']
    
    # COSTS
    ordered_price = (df_b
                 .select(OD_id+ ['trip_cost', 'RecordID'])
                 .sort(OD_id+ ['trip_cost'], descending=False)
                 .with_columns(pl.Series(values=np.arange(df_b.select( ['RecordID', 'trip_cost']).collect().shape[0]), name='new_col')
                               .alias('index')
                               .cast(pl.Int32)))
    reperes = (ordered_price.lazy().groupby(OD_id)
            .agg([pl.min('index').suffix('_n_min'), 
                    pl.max('index').suffix('_n_max')]))
    complete_df = (ordered_price.join(reperes, on=OD_id))
    q_cost = complete_df.with_columns(((pl.col('index')-pl.col('index_n_min'))/(pl.max(pl.col('index_n_max')-pl.col('index_n_min'), 1))).alias('cost_quantile')).select(OD_id + ['cost_quantile', 'RecordID'])
    cost_span = (df_b
             .select(OD_id+ ['trip_cost', 'Max_trip_cost', 'Min_trip_cost', 'RecordID'])
             .sort(OD_id+ ['trip_cost'], descending=False)
             .with_columns( ((pl.col('trip_cost')- pl.col('Min_trip_cost'))/pl.max(pl.col('Max_trip_cost')- pl.col('Min_trip_cost'), 1))
                           .alias('cost_span_q').cast(pl.Float64))
             .with_columns(((pl.col('Max_trip_cost')- pl.col('Min_trip_cost'))/pl.col('Min_trip_cost'))
                           .alias('cost_span_ratio'))
             )        
    cost_relative_infos = cost_span.join(q_cost, on=OD_id+['RecordID']).drop(['Max_trip_cost', 'Min_trip_cost', 'trip_cost'])
    
    # FARES
    fare_OD_infos = (df_b
                 .select(OD_id+ ['fare_predicted', 'RecordID'])
                 .groupby(OD_id)
                     .agg([pl.mean('fare_predicted').alias('fare_predicted_mean_OD'),
                           pl.std('fare_predicted', ddof=0).alias('fare_predicted_std_OD'),
                           pl.min('fare_predicted').alias('fare_predicted_min_OD'),
                           pl.max('fare_predicted').alias('fare_predicted_max_OD'),
                           ])
                     )

    # PRESENCE
    df_pres= df_b.select(OD_id + ['RecordID', 'available_seats1', 'available_seats2', 'stops_number'])
    weight_0stop = WEIGHT_0STOPS
    weight_1stop = WEIGHT_1STOPS 
    weighted_seats_df = (df_pres.with_columns(pl.col('available_seats2').fill_null(pl.col('available_seats1')))
        .with_columns(
            pl.min(pl.col('available_seats1'), pl.col('available_seats1'))
            .alias('max_IT_available_seats'))
        .drop(columns=['available_seats1', 'available_seats2'])
        .with_columns(
            (pl.col('max_IT_available_seats')*(pl.col('stops_number')*weight_1stop + (1-pl.col('stops_number'))*weight_0stop ))
            .alias('weighted_available_seats'))
        .with_columns(
            (pl.col('max_IT_available_seats')*(pl.col('stops_number')))
            .alias('seats_1stop'))
        .with_columns(
            (pl.col('max_IT_available_seats')*(1-pl.col('stops_number')))
            .alias('seats_0stop'))
        )

    weighted_presence = (weighted_seats_df
                        .groupby(OD_id)
                        .agg([pl.sum('weighted_available_seats').alias('weighted_OD_seats'),
                            pl.sum('max_IT_available_seats').alias('weight_absolute_OD_seats'),
                            pl.sum('seats_1stop').alias('total_seats_1stop'),
                            pl.sum('seats_0stop').alias('total_seats_0stop'),
                            ])
                        .join(weighted_seats_df, OD_id)
                        .with_columns((pl.col('weighted_available_seats')/pl.col('weighted_OD_seats')).alias('weighted_mkshare'))
                        .with_columns((pl.col('seats_1stop')/pl.max(pl.col('total_seats_1stop'), 1)).alias('1stop_mkshare'))
                        .with_columns((pl.col('seats_0stop')/pl.max(pl.col('total_seats_0stop'), 1)).alias('0stop_mkshare'))
                        .drop(columns=['weighted_OD_seats', 'weighted_available_seats', 'max_IT_available_seats'])
                        )

    weighted_HHI = (weighted_presence
                    .with_columns(
                        pl.col('weighted_mkshare')
                        .apply(lambda x: x**2)
                        .alias('weighted_mkshare_sq'))
                    .groupby(OD_id).agg([pl.sum('weighted_mkshare_sq').alias('weighted_HHI_OD')])
                    )

    competition_carr_type= (df_b.select(OD_id + ['RecordID', 'carrier_type_new']).groupby(OD_id + ['carrier_type_new']).agg([(1/pl.count('RecordID')).alias('carr_type_flights_nb_ind')]))
    competition_OD = df_b.select(OD_id + ['RecordID']).groupby(OD_id).agg([(1/pl.count('RecordID')).alias('total_flights_nb_ind')])
    competition = competition_carr_type.join(competition_OD, OD_id)

    weighted_presence2 = (weighted_presence
                        .join(df_b.select(['carrier_type_new', 'RecordID']), 'RecordID' )
                        .join(competition, OD_id + ['carrier_type_new'])
                        .join(weighted_HHI, OD_id)
                        .drop(columns=['carrier_type_new','stops_number'])
                        )
    
    indicators = (weighted_presence2.join(cost_relative_infos, on=OD_id + ['RecordID'])).join(fare_OD_infos, on=OD_id)
    
    cols =[
    #  'available_seats1',
    #  'available_seats2',
    #  'carrier',
    'carrier_type_new',
    'count_1stop_itineraries_on_OD',
    'count_nonstop_itineraries_on_OD',
    'dest',
    #  'dest_city',
    #  'dest_country_code',
    #  'dest_lat',
    #  'dest_long',
    'distance_detour_factor',
    'distance_km_flown',
    'distance_km_OD',
    #  'distance_km_OD_squared',
    #  'fare',
    'fare_index',
    'first_stop',
    #  'first_stop_country_code',
    #  'first_stop_region',
    #  'helper_MIN_seats',
    'itinerary_time',

    #  'Max_trip_cost',
    #  'Min_itinerary_time',
    #  'Min_trip_cost',
    'orig',
    #  'orig_city',
    #  'orig_country_code',
    #  'orig_lat',
    #  'orig_long',
    'pax_share',
    'quarter',
    #  'quarter_year',
    'RecordID',
    'seat_density_factor',
    'stops_number',
    'time_detour_factor',
    #  'training_data_fare',
    'training_data_paxshare',
    'trip_cost',
    'trip_cost_vs_highest_cost_on_OD',
    'trip_cost_vs_lowest_cost_on_OD',
    'year',
    #  'first_stop_city',
    'market_share_seats_0stop',
    #  'market_share_seats-1stop_proxy',
    #  'helper_ms_0stop',
    #  'helper_ms_1stop',
    #  'market_share_0stop1stop_proxy',
    'HHI_OD_0stop',
    'HHI_OD_1stop',
    'fare_predicted'
    ]
    df_b2 = df_b.select(cols).join(indicators, on=OD_id + ['RecordID']).with_columns((1/pl.col('total_flights_nb_ind')).alias('weight_flight_nb'))
    
    
    return df_b2.collect()
    


def lambda_handler(event, context):
    # get first file from S3
    bucket_1 = 'revenue-input-2'
    key_1 = 'Demand_DATA_STEP4_B1_fare_python_input_v11.csv'
    response_1= s3.get_object(Bucket=bucket_1, Key=key_1)
    # Read in S3 data
    csv_content1 = response_1['Body'].read()
    df_input_data = pd.read_csv(io.BytesIO(csv_content1))
    
    
    # get second file from S3
    bucket_2 = 'revenue-output-2'
    key_2 = 'Demand_DATA_STEP5_B1_fare_ml_output_v11.csv'
    response_2= s3.get_object(Bucket=bucket_2, Key=key_2)
    # Read in S3 data
    csv_content2 = response_2['Body'].read()
    df_output_picked = pd.read_csv(io.BytesIO(csv_content2))

    # build output
    otp = transform_data(df_input_data, df_output_picked)
    # otp is a ploars dataframe

    
    # Convert DataFrame to CSV data in memory
    csv_buffer = io.StringIO()
    otp.write_csv(csv_buffer)
    
    # Define your S3 bucket and object key
    bucket_name = 'revenue-input-2'
    object_key = 'Demand_DATA_STEP5_B1_fare_ml_output_v11_transformed.csv'

    # Upload the CSV data to S3
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())


    return {
        'statusCode': 200,
        'body': 'Data preparation completed'
    }
    
    