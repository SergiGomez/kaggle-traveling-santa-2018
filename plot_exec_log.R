# This script plots the score of the different tsp processes to see how they improve with time

library(data.table)
library(ggplot2)
library(foreach)
path <- "~/Documents/Data_Science/santa/"
file <- "log_score.txt"

subfolders <- c("1214_LKH_local/", "1214_LKH_2_local/", "1214_LKH_3_local/", "1214_LKH_4_local/",
                "1214_LKH_MC_local/", "1214_LKH_MC2_local/", "1219_LKH_MT8_AWS/", "1219_LKH_MT10_AWS/",
                "1219_LKH_MT12_AWS/", "1219_LKH_MT8init_AWS/", "1219_LKH_MT14_AWS/", "1222_LKH_PA2C1_init_AWS/",
                "1222_LKH_MT10_init_AWS/")

data.list <- foreach(subfolder = subfolders) %do% {
  if (file.exists(paste0(path,subfolder,file))){
    data <- fread(paste0(path,subfolder,file))
    setnames(data, c("date", "seconds", "score"))
    data[, exec := subfolder]
    data[, time_diff := (seconds - shift(seconds))/3600]
    data <- data[2:.N]
    data[, score := as.numeric(score)]
  }
}

data <- rbindlist(data.list)
data[, time := cumsum(time_diff), exec]

ggplot(data, aes(x = time, y = score, color = exec)) +
  geom_point() +
  xlim(c(0,300))

# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1219_LKH_MT8_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1219_LKH_MT10_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1219_LKH_MT12_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1219_LKH_MT14_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1219_LKH_MT8init_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1222_LKH_PA2C1_init_AWS /home/luis/Documents/Data_Science/santa
# scp -ri ~/Documents/keys/ubuntu_to_aws.pem ubuntu@ec2-18-222-166-159.us-east-2.compute.amazonaws.com:~/santa/1222_LKH_PA2C1_AWS /home/luis/Documents/Data_Science/santa
