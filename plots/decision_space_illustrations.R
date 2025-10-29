library(readxl)
library(scales)
library(ggplot2)
library(colorspace)
library(tidyverse)
library("svglite")

# functions
# Source the external script to load functions
source("decision_space_functions.R")

# read output file
# input_name <- "single_run_25_nodes_2024_09_25_15_52"
input_name <- "exp3"
input_file <- paste0(input_name,".xlsx")

# Step 2: Create a folder in the current directory using the input file's name
output_directory <- file.path(getwd(), input_name)  # Create the full path
if (!dir.exists(output_directory)) {
  dir.create(output_directory)  # Create the directory if it doesn't exist
}

# Get the names of all sheets in the Excel file
sheet_names <- excel_sheets(input_file)

# Initialize an empty list to store the data frames
df_list <- list()

# Loop through the sheet names and read each sheet
for (sheet in sheet_names) {
  # Read the current sheet and store it as a data frame in the list
  df_list[[sheet]] <- read_excel(input_file, sheet = sheet)
}

# Read inputs
inputs_problem_data_df <- df_list[["inputs_problem_data"]]
node_coordinates <- inputs_problem_data_df[, c("node_id", "x_coordinate", "y_coordinate")]



# problem parameters
base_id <- 1
water_id <- inputs_problem_data_df$node_id[inputs_problem_data_df$node_state == 5]
node_at_a_side <- sqrt(nrow(inputs_problem_data_df))
pX <- inputs_problem_data_df$x_coordinate 
pY <- inputs_problem_data_df$y_coordinate
pMat <- c()


# vehicle routes
x_ijk_results <- df_list[["x_ijk_results"]]
# Filter rows where 'value' == 1
x_ijk_results <- x_ijk_results[x_ijk_results$value == 1, ]

# Extract the value of 'n_vehicles'
n_vehicles <-  as.numeric(df_list[["inputs_parameters"]]$value[ df_list[["inputs_parameters"]]$parameter == "n_vehicles"])

# Initialize a list to store the routes for each vehicle
vehicle_routes <- list()

# Loop through vehicle_ids from 1 to n_vehicles
for (vehicle_id in 1:n_vehicles) {
  # Extract the route for the current vehicle_id
  route <- as.numeric(extract_route(vehicle_id, x_ijk_results, water_id))
  
  # Store the route in the list using the vehicle_id as the key
  vehicle_routes[[as.character(vehicle_id)]] <- route
}


# node arrival times
tv_j_results <- round_df(df_list[["tv_j_results"]],2)
t_v_list <- tv_j_results$value




# Create an empty list to store the scenarios
scenario_data_list <- list()

# each of these results data frames has a column 'scenario_id' and 'node_id'
# Step 1: Get the unique scenario IDs (assuming the same scenarios exist in all result sets)
scenario_ids <- unique(df_list[["inputs_scenario_rates"]]$scenario_id)

# Read the second sheet into a data frame
y_jw_results <- df_list[["y_jw_results"]]
ts_jw_results <- round_df(df_list[["ts_jw_results"]],2)
tm_jw_results <- round_df(df_list[["tm_jw_results"]],2)
te_jw_results <- round_df(df_list[["te_jw_results"]],2)
r_jw_results <- round_df(df_list[["r_jw_results"]],2)

# Step 2: Loop through each scenario and construct a data frame for each
for (scenario_id in scenario_ids) {
  
  # Extract the relevant rows for the current scenario from each results data frame
  y_jw <- y_jw_results[y_jw_results$scenario_id == scenario_id, ]
  ts_jw <- ts_jw_results[ts_jw_results$scenario_id == scenario_id, ]
  tm_jw <- tm_jw_results[tm_jw_results$scenario_id == scenario_id, ]
  te_jw <- te_jw_results[te_jw_results$scenario_id == scenario_id, ]
  r_jw <- r_jw_results[r_jw_results$scenario_id == scenario_id, ]
  
  # each result data frame has 'node_id' and 'value' columns, combine them
  scenario_data <- data.frame(
    node_id = y_jw$node_id,          # Assuming all frames have a common node_id
    y_jw = y_jw$value,               # y_jw attribute
    ts_jw = ts_jw$value,             # ts_jw attribute
    tm_jw = tm_jw$value,             # tm_jw attribute
    te_jw = te_jw$value,             # te_jw attribute
    r_jw = r_jw$value                # r_jw attribute
  )
  
  # Store the scenario data frame in the list with the scenario ID as the key
  scenario_data_list[[as.character(scenario_id)]] <- scenario_data
}


# NEW: value & spread maps (save before initial fires)
plot_value_map(inputs_problem_data_df, output_directory)
plot_spread_map(inputs_problem_data_df, output_directory)  # use_rate=TRUE by default

plot_scenarios(inputs_problem_data_df, vehicle_routes, scenario_data_list, output_directory)

plot_scenarios(inputs_problem_data_df, vehicle_routes, scenario_data_list, output_directory, include_routes = FALSE)

plot_initial_fires(inputs_problem_data_df, output_directory)

