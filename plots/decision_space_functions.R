library(readxl)
library(scales)
library(ggplot2)
# Function to round all numeric columns in a data frame
round_df <- function(df, decimals = 2) {
  df[] <- lapply(df, function(x) if(is.numeric(x)) round(x, decimals) else x)
  return(df)
}

# functions
# plot vehicle route
plot_vehicle_route <- function(visiting_order, vehicle_col, vehicle_line_type) {
  transparent_col <- adjustcolor(vehicle_col, alpha.f = 0.7)  # Alpha between 0 (fully transparent) and 1 (fully opaque)
  
  for (i in 1:(length(visiting_order)-1)) {
    arrows(pX[visiting_order[i]],pY[visiting_order[i]], pX[visiting_order[i+1]],pY[visiting_order[i+1]], col=transparent_col,lwd=3, lty=vehicle_line_type, length = 0.25, angle = 20)
    i=i+1
  }
}

# plot nodes
plot_node_add <- function(status, node_x, node_y, node_id, t_s, t_m, t_e, r_j, t_v = NULL, col_def){
  if (status == "clean") {
    col_to_use = "white"
    points(x=node_x, y=node_y, col = "black", bg = alpha(col_to_use, 1),  pch = 21, cex = 3.5)
    
  }
  else if(status == "visited"){
    col_to_use = "forestgreen"
    # text(node_x, node_y + 0.25,labels =  bquote(t[.(node_id)]^v == .(t_v)), col = col_to_use, cex = 0.5)
    text(node_x, node_y + 0.25,labels =  bquote(t == .(t_v)), col = col_to_use, cex = 0.5)
    
  } else if(status == "burned_out"){
    col_to_use = "red"
    text(node_x, node_y + 0.25,labels =  bquote(t == .(t_v)), col = col_to_use, cex = 0.5)
    
  }
  else{
    col_to_use = "red"
    # text(node_x, node_y + 0.25,labels =  bquote(t == .(t_v)), col = col_to_use, cex = 0.5)
    
  }
  points(x=node_x, y=node_y, col = "black", bg = alpha(col_to_use, 0.3),  pch = 21, cex = 3.5)
  text(x=node_x, y=node_y, col = "black", labels = node_id, cex = 1)
  text(node_x, node_y + 0.38,labels =  bquote(tw == "[" *.(t_s) *"," *.(t_m) *","* .(t_e) *"]"), col = col_to_use, cex = 0.5)
  text(node_x, node_y - 0.38,labels =  bquote(r == .(r_j)), col = col_to_use, cex = 0.5)
  
}
# Define a function to extract the route for a specific vehicle_id
extract_route2 <- function(vehicle_id, x_ijk_results, water_id) {
  # Filter the data for the specific vehicle_id
  x_ijk_results_vehicle <- x_ijk_results[x_ijk_results$vehicle_id == vehicle_id, ]
  
  # Initialize the route with the starting node (1)
  route <- c(1)
  
  # Start at node 1 and find the route
  current_node <- 1
  
  # Create a set to track visited nodes to avoid infinite loops
  visited_nodes <- c()
  
  while(TRUE) {
    
    # Add the current node to the visited list
    visited_nodes <- c(visited_nodes, current_node)
    
    # Find the next node in the sequence
    # If the current node is an intermediate node, we allow revisiting it multiple times
    if (current_node %in% water_id) {
      next_row <- x_ijk_results_vehicle[x_ijk_results_vehicle$from_node_id == current_node & x_ijk_results_vehicle$to_node_id != current_node, ]
    } else {
      next_row <- x_ijk_results_vehicle[x_ijk_results_vehicle$from_node_id == current_node & !(x_ijk_results_vehicle$to_node_id %in% visited_nodes), ]
    }
    # # Find the next node in the sequence
    # next_row <- x_ijk_results_vehicle[x_ijk_results_vehicle$from_node_id == current_node, ]
    # 
    if (nrow(next_row) == 0) {
      break  # Exit the loop when no more connections are found
    }
    
    # Get the next node (to_node_id)
    next_node <- next_row$to_node_id[1]
    
    # Add the next node to the route
    route <- c(route, next_node)
    
    # Move to the next node
    current_node <- next_node
    
    # If we return to the starting node (1), stop
    if (current_node == 1) {
      break
    }
  }
  # Ensure the final return to node 1 is included if missing
  if (route[length(route)] != 1) {
    route <- c(route, 1)
  }
  return(route)
}

extract_route <- function(vehicle_id, x_ijk_results, water_id) {
  
  # Filter the data for the specific vehicle_id
  x_ijk_results_vehicle <- x_ijk_results[x_ijk_results$vehicle_id == vehicle_id, ]
  
  
  # Remove rows where 'from_node_id' or 'to_node_id' are in water_id
  filtered_df <- x_ijk_results_vehicle[!(x_ijk_results_vehicle$from_node_id %in% water_id |
                                           x_ijk_results_vehicle$to_node_id %in% water_id), ]
  
  # Initialize the route with the first 'from_node_id'
  route <- c(filtered_df$from_node_id[1])
  
  # Set the current node to the first node
  current_node <- filtered_df$from_node_id[1]
  
  while (TRUE) {
    # Find the row where 'from_node_id' equals 'current_node'
    next_row <- filtered_df[filtered_df$from_node_id == current_node & 
                              filtered_df$value == 1, ]
    
    # If no more moves are found, stop the loop
    if (nrow(next_row) == 0) break
    
    # Get the 'to_node_id' from the row found
    next_node <- next_row$to_node_id[1]
    
    # Add the next node to the route
    route <- c(route, next_node)
    
    # Update the 'current_node' to the 'to_node_id'
    current_node <- next_node
    
    # Remove the processed row to avoid looping
    filtered_df <- filtered_df[-which(filtered_df$from_node_id == next_row$from_node_id[1] &
                                        filtered_df$to_node_id == next_row$to_node_id[1]), ]
    
    # If we have looped back to the starting node, stop
    if (current_node == route[1]) break
  }
  
  # Filter rows where 'to_node_id' is in 'water_id'
  intermediate_nodes_df <- x_ijk_results_vehicle[x_ijk_results_vehicle$to_node_id %in% water_id, ]
  
  # Initialize an extended route vector
  extended_route <- c()
  
  # Loop over each node in the main route
  for (i in seq_along(route)) {
    # Add the current main node to the extended route
    current_node <- route[i]
    extended_route <- c(extended_route, current_node)
    
    # Check if the current node is in 'from_node_id' of intermediate_nodes_df
    intermediate_rows <- intermediate_nodes_df[intermediate_nodes_df$from_node_id == current_node, ]
    
    # If there are any intermediate nodes after this main node, add them
    if (nrow(intermediate_rows) > 0) {
      # Add the intermediate 'to_node_id' values to the extended route
      extended_route <- c(extended_route, intermediate_rows$to_node_id)
    }
  }
  
  
  return(extended_route)
}






# Function to draw diagonal hatching lines within a rectangle
draw_hatching <- function(node_x, node_y, step = 0.1, node_type) {
  
  if (node_type == "block") {
    color="gray80"
    lty_type = 1
    lwd_width = 1
  }else {
    color = "deepskyblue"
    lty_type = 2
    lwd_width = 2
    
  }
  # Draw diagonal lines from bottom-left to top-right
  for (offset in seq(0, 1, by = step)) {
    # Diagonal from bottom-left to top-right
    segments(x0 = node_x - 0.5, y0 = node_y - 0.5 + offset, 
             x1 = node_x - 0.5 + offset, y1 = node_y - 0.5, col = color, lty = lty_type, lwd=lwd_width)
    
    # Diagonal from top-left to bottom-right
    segments(x0 = node_x - 0.5 + offset, y0 = node_y + 0.5, 
             x1 = node_x + 0.5, y1 = node_y - 0.5 + offset, col = color, lty = lty_type, lwd=lwd_width)
  }
}

# Example function to plot the base grid and vehicle routes
plot_base_grid <- function(inputs_problem_data_df) {
  # Adjust the margins for all plots (to give more room for axes and title)
  par(pty = "s",mar = c(5, 5, 5, 2) + 0.1)  # Bottom, left, top, right margins
  
  plot(c(), xlim=c(0,ceiling(max(pX))), ylim=c(0,ceiling(max(pY))),asp=1, xaxs="i", yaxs="i",xaxt='n', yaxt="n", ylab="", xlab="")
  # plot(c(), xlim=c(0,ceiling(max(pX))), ylim=c(0,ceiling(max(pY))),asp=1,xaxs="i", yaxs="i",xaxt='n', yaxt="n", ylab="", xlab="", bty="n") # see ?plot.function
  x <- seq(0, ceiling(max(pX)), 1)
  axis(side = 1, at = x, labels = T)
  axis(side = 2, at = x, labels = T)
  grid(nx = node_at_a_side, ny = node_at_a_side, col = "gray25", lty = "dotted",lwd = par("lwd"), equilogs = TRUE)
  
  # add base
  points(x=0.5,y=0.5,col = "black", pch=17,cex=3)
  
  node_coordinates <- inputs_problem_data_df[, c("node_id", "x_coordinate", "y_coordinate")]
  text(x=node_coordinates$x_coordinate , y=node_coordinates$y_coordinate, col = "gray40", labels = node_coordinates$node_id, cex = 0.8)
  
  block_nodes <- inputs_problem_data_df$node_id[inputs_problem_data_df$node_state == 4]
  block_node_df<- inputs_problem_data_df[block_nodes, c("node_id", "x_coordinate", "y_coordinate") ]
  
  # Loop through each row in block_node_df to draw rectangles and hatching for each blocked node
  for (i in 1:nrow(block_node_df)) {
    # Extract the x and y coordinates of the blocked node
    node_x <- block_node_df$x_coordinate[i]
    node_y <- block_node_df$y_coordinate[i]


    # Add diagonal hatching lines
    draw_hatching(node_x, node_y, node_type="block")
    
   }
  # text(0.5,0.5-0.2,labels = bquote(t[s] == 0), col=1)
  # text(0.5,0.5-0.2,labels = bquote(t[h] == .(tail(t_v_list,1))), col=1)
  # Reset the margins and plot type to default after plotting
  par(mar = c(5, 4, 4, 2) + 0.1, pty = "m")
}

plot_water_resources <- function(water_id) {
  water_resouce_nodes = water_id
  for (i in water_resouce_nodes) {
    draw_hatching(pX[water_id], pY[water_id], node_type = "water")
    # points(x=pX[water_id],y=pY[water_id],col = "black", bg=alpha("deepskyblue", 0.3), pch=21,cex=3.5)
    text(x=pX[water_id],y=pY[water_id],col = "black", labels=water_id ,cex=1)
  }
}

# Example function to plot the base grid and vehicle routes
plot_routes <- function(vehicle_routes) {
  # add routes
  #colors <- dark_hcl(n_vehicles)  # Generates a range of distinct colors
  colors <- qualitative_hcl(n_vehicles, palette = "Dark 3")
  
  # line_types <- 1:n_vehicles      # Assign different line types (from 1 to n_vehicles)
  line_types <- rep(1, n_vehicles)      # Assign different line types (from 1 to n_vehicles)
  
  # Loop through each vehicle's route and plot using the assigned color and line type
  for (i in 1:n_vehicles) {
    vehicle_id <- names(vehicle_routes)[i]  # Get the vehicle ID (key in the list)
    route <- vehicle_routes[[vehicle_id]]   # Get the vehicle route (vector of nodes)
    
    # Assign a color and line type for the vehicle
    color <- colors[i]
    line_type <- line_types[i]
    
    # Call the plot_vehicle_route function for the current vehicle
    plot_vehicle_route(route, color, line_type)
  }
}



# add node results for the routes
plot_node_attributes <- function(scenario_data, include_routes = TRUE) {
  # Plot attributes for each node (e.g., color, size, label)
  
  nodes_with_fire <- as.numeric(scenario_data$node_id[scenario_data$y_jw > 0])
  
  # add nodes that had fire
  for (i in nodes_with_fire){
    node_row <- scenario_data[scenario_data$node_id == i, ]
    plot_node_add("nodes_with_fire", pX[i],  pY[i], i, node_row$ts_jw, node_row$tm_jw, node_row$te_jw, node_row$r_jw, "gray20") # node 3
  }
  
  if (include_routes == TRUE) {
  # nodes_visited <- as.numeric(scenario_data$node_id[scenario_data$y_jw > 0])
  # v<-vehicle_routes[[2]]
  for (v in vehicle_routes) {
    for (i in setdiff(unique(v), c(base_id, water_id))) {
      #i=25
      node_row <- scenario_data[scenario_data$node_id == i, ]
      
      arrival_time <- tv_j_results$value[tv_j_results$node_id == i]
      
      # Check if arrival_time is between start and end of fire and if fire exists
      if (node_row$y_jw ==1  & arrival_time >= node_row$ts_jw & arrival_time < node_row$te_jw) {
        plot_node_add("clean", pX[i], pY[i], i, node_row$ts_jw, node_row$tm_jw, node_row$te_jw, node_row$r_jw, "gray20") 
        plot_node_add("visited", pX[i], pY[i], i, node_row$ts_jw, node_row$tm_jw, node_row$te_jw, node_row$r_jw, arrival_time, "gray20")
      }else if(node_row$y_jw ==1  & arrival_time < node_row$ts_jw | arrival_time >= node_row$te_jw){
        plot_node_add("burned_out", pX[i], pY[i], i, node_row$ts_jw, node_row$tm_jw, node_row$te_jw, node_row$r_jw, arrival_time, "gray20")
        
      } 
    }
  }
  }
}


# Example main function to plot grid and routes for each scenario
plot_scenarios <- function(inputs_problem_data_df, vehicle_routes, scenario_data_list, output_directory, include_routes=TRUE) {
  
  # Determine how many scenarios we have
  n_scenarios <- length(scenario_data_list)
  
  water_id <- inputs_problem_data_df$node_id[inputs_problem_data_df$node_state == 5]
  
  
  # Adjust the margins for all plots (to give more room for axes and title)
  par(mar = c(5, 5, 2, 2) + 0.1)  # Bottom, left, top, right margins
  
  
  # Loop through each scenario and plot the grid and node attributes
  for (scenario_id in 1:n_scenarios) {
    # Set up the file name for saving
    if (include_routes == TRUE) {
      file_name <- paste0(output_directory, "/scenario_", scenario_id, ".png")
      file_name_svg <- paste0(output_directory, "/scenario_", scenario_id, ".svg")
      
    } else {
      file_name <- paste0(output_directory, "/fire_spread_scenario_", scenario_id, ".png")
      file_name_svg <- paste0(output_directory, "/fire_spread_scenario_", scenario_id, ".svg")
      
    }
    
    # Define your plotting logic as a function
    plot_graph <- function() {
      # Start a new plot for each scenario
      plot_base_grid(inputs_problem_data_df)
      
      plot_water_resources(water_id)
      
      
      # Get the scenario-specific data
      scenario_data <- scenario_data_list[[scenario_id]]
      
      title(main = paste("Scenario", scenario_id, ":",  round(sum(scenario_data$r_jw),2)), line=2, cex.main=1)
      
      
      if (include_routes == TRUE) {
        # Overlay node attributes for the current scenario
        plot_node_attributes(scenario_data)
        
        plot_routes(vehicle_routes)
        
        # Add a title indicating the scenario number
        legend_text = c()
        for (v in (1:length(vehicle_routes))){
          legend_text <- append(legend_text, paste0("UAV ", v, ": ", paste(vehicle_routes[[v]], collapse = "-")))
        }
        
        colors <- qualitative_hcl(n_vehicles, palette = "Dark 3")
        line_types <- 1:n_vehicles      # Assign different line types (from 1 to n_vehicles)
        legend('top', legend_text, pch = c(NA, NA), lty = line_types, col = colors, 
               lwd=c(2,2),text.col = colors, cex = 1, 
               xpd=TRUE, bty = "n", inset=c(0, -0.13),
               ncol = 1)
        
        
      } else {
        plot_node_attributes(scenario_data, include_routes=FALSE)
      }
    }
    
    # Save the plot to a PNG file
    png(file_name, width = 1200, height = 1200, res = 200)
    # Call the plotting function
    plot_graph()
    
    # Close the PNG device
    dev.off()
    
    cat("Saved Scenario PNG", scenario_id, "to", file_name, "\n")
    
    
    
    # Save the plot to an SVG file
    # svg(file_name_svg, width = 1200, height = 1200)
    # Call the same plotting function
    svg(file_name_svg)
    
    # Code of the plot
    plot_graph()
    
    # Close the SVG device
    dev.off()
    
    
    # 
    # # Close the SVG device
    # dev.off()
    # 
      
    # Notify the user that the plot has been saved
    cat("Saved Scenario SVG", scenario_id, "to", file_name_svg, "\n")
    
  }
  
  # Reset the margins to default after plotting
  par(mar = c(5, 4, 4, 2) + 0.1)
} 



# Example main function to plot grid and routes for each scenario
plot_scenarios_only_fires <- function(scenario_data_list,  output_directory) {
  
  # Determine how many scenarios we have
  n_scenarios <- length(scenario_data_list)
  
  # Adjust the margins for all plots (to give more room for axes and title)
  par(mar = c(8, 5,4 , 2) + 0.1)  # Bottom, left, top, right margins
  
  
  # Loop through each scenario and plot the grid and node attributes
  for (scenario_id in 1:n_scenarios) {
    
    # Set up the file name for saving
    file_name <- paste0(output_directory, "/fire_spread_scenario_", scenario_id, ".png")
    file_name_svg <- paste0(output_directory, "/fire_spread_scenario_", scenario_id, ".svg")
    
    plot_graph <- function() {
      # Start a new plot for each scenario
      plot_base_grid()
      
      plot_water_resources(water_id)
      
      # Get the scenario-specific data
      scenario_data <- scenario_data_list[[scenario_id]]
      
      # Overlay node attributes for the current scenario
      plot_node_attributes(scenario_data, include_routes=FALSE)
      
      
      # Add a title indicating the scenario number
      title(main = paste("Scenario", scenario_id), line=2, cex.main=1)
      
    }
    
    # Save the plot to a PNG file
    png(file_name, width = 1200, height = 1200, res = 200)
    # Call the plotting function
    plot_graph()
    
    # Close the PNG device
    dev.off()
    
    # Notify the user that the plot has been saved
    cat("Saved Scenario PNG", scenario_id, "to", file_name, "\n")
    
    # svg(file_name_svg, width = 1200, height = 1200)
    # Call the same plotting function
    svg(file_name_svg)
    
    # Code of the plot
    plot_graph()
    
    # Close the SVG device
    dev.off()
    
    # Close the SVG device
    #dev.off()
  

    cat("Saved Scenario SVG", scenario_id, "to", file_name_svg, "\n")
  }
  
  # Reset the margins to default after plotting
  par(mar = c(5, 4, 4, 2) + 0.1)
} 


plot_initial_fires <- function(inputs_problem_data_df, output_directory){
  # Adjust the margins for all plots (to give more room for axes and title)
  par(mar = c(8, 5, 4, 2) + 0.1)  # Bottom, left, top, right margins
  
  # Set up the file name for saving
  file_name <- paste0(output_directory, "/fire_spread_scenario_0_initial_fires", ".png")
  file_name_svg <- paste0(output_directory, "/fire_spread_scenario_0_initial_fires", ".svg")
  
  # Define your plotting logic as a function
  plot_graph <- function() {
    # Start a new plot for each scenario

    plot_base_grid(inputs_problem_data_df)


    water_id <- inputs_problem_data_df$node_id[inputs_problem_data_df$node_state == 5]
    
    plot_water_resources(water_id)
    
    nodes_with_initial_fires <- as.numeric(inputs_problem_data_df$node_id[inputs_problem_data_df$node_state == 1])
    # Get the scenario-specific data
    
    # # add nodes that had fire at start
    # for (i in nodes_with_initial_fires){
    #   node_row <- scenario_data[scenario_data$node_id == i, ]
    #   plot_node_add("nodes_with_fire", pX[i],  pY[i], i, node_row$ts_jw, node_row$tm_jw, node_row$te_jw, node_row$r_jw, "gray20") # node 3
    # }
    
    # add nodes that had fire at start
    for (i in nodes_with_initial_fires){
      node_row <- inputs_problem_data_df[inputs_problem_data_df$node_id == i, ]
      plot_node_add("nodes_with_fire", node_row$x_coordinate,  node_row$y_coordinate, i, 0, round(1/node_row$fire_spread_rate,2), round((1/node_row$fire_spread_rate + 1/node_row$fire_amelioration_rate),2), node_row$initial_value, "gray20") # node 3
    }
    
    # Add a title indicating the scenario number
    title(main = "initial fires", line=2, cex.main=1)
    
  }
  
  
  # Save the plot to a PNG file
  png(file_name, width = 1200, height = 1200, res = 200)
  # Call the plotting function
  plot_graph()
  
  # Close the PNG device
  dev.off()
  cat("Saved initial fires PNG", "to", file_name, "\n")
  
  
  
  #svg(file_name_svg, width = 1200, height = 1200)
  # Call the same plotting function
  # SVG graphics device
  svg(file_name_svg)
  
  # Code of the plot
  plot_graph()
  
  # Close the SVG device
  dev.off()
 
  # Notify the user that the plot has been saved
  cat("Saved initial fires SVG", "to", file_name_svg, "\n")
  
  # Reset the margins to default after plotting
  par(mar = c(5, 4, 4, 2) + 0.1)
}
