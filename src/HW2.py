# HW2 - F1 Pit Stop Analysis
# ===========================

from pyspark.sql.functions import (
    concat_ws, datediff, to_date, floor, col, avg, min, max,
    when, upper, substring, year, sum, round
)
from pyspark.sql.window import Window

# ===========================
# Load Data
# ===========================

df_pitstops = spark.read.csv("/Volumes/gr5069/raw/f1_data/pit_stops.csv", header=True)
drivers = spark.read.csv("/Volumes/gr5069/raw/f1_data/drivers.csv", header=True)
driver_standing = spark.read.csv("/Volumes/gr5069/raw/f1_data/driver_standings.csv", header=True)
df_races = spark.read.csv("/Volumes/gr5069/raw/f1_data/races.csv", header=True)
race_results = spark.read.csv("/Volumes/gr5069/raw/f1_data/results.csv", header=True)


# ===========================
# 1.1 Average time each driver spent at the pit stop for each race
# ===========================


df = df_pitstops.withColumn("milliseconds", col("milliseconds").try_cast("int"))
driver_avg = df.groupBy("raceId", "driverId").agg(
    floor(avg("milliseconds")).alias("avg_pit_time"),
).orderBy("raceId")

display(driver_avg.head(5))


# ===========================
# 1.2 Slowest and Fastest Pit Stop in Each Race
# ===========================


race_stats = df.groupBy("raceId").agg(
    min("milliseconds").alias("fastest_pit_stop in ms"),
    max("milliseconds").alias("slowest_pit_stop in ms"),
).orderBy("raceId")

display(race_stats.head(5))


# ===========================
# Drivers Pit stop average with drivers name
# ===========================


drivers_pitstops = driver_avg.join(
    drivers.select("driverId", "forename", "surname"),
    on="driverId"
).orderBy("raceId")

display(drivers_pitstops.head(5))


# ===========================
# 2. Rank order by finishing position the average time spent at the pit stop in each race
# ===========================

pitstop_position = drivers_pitstops.join(
    driver_standing.select("raceId", "driverId", "position"),
    on=["raceId", "driverId"]
).withColumn("position", col("position").cast("int")).orderBy("position", "avg_pit_time")

display(pitstop_position.head(5))


# ===========================
# 3. Insert the missing code (e.g: ALO for Alonso) for drivers based on the 'drivers' dataset
# ===========================

df_drivers_fixed = drivers.withColumn(
    "code",
    when(col("code") == "\\N", upper(substring(col("surname"), 1, 3)))
    .otherwise(col("code"))
)

display(df_drivers_fixed.head(5))

pit_avg_by_finish_pos = pitstop_position.join(
    df_drivers_fixed.select("driverId", "code", "dob"),
    on="driverId"
).orderBy("driverId")

display(pit_avg_by_finish_pos.head(5))


# ===========================
# 4. Who is the youngest and the oldest driver in each race?
# ===========================

# concat the race date to the df
pit_avg_by_finish_pos_with_age = pit_avg_by_finish_pos.join(
    df_races.select("raceId", col("date").alias("race_date")),
    on="raceId"
)

pit_avg_by_finish_pos = pit_avg_by_finish_pos_with_age.withColumn(
    "age",
    floor(datediff(to_date(col("race_date")), to_date(col("dob"))) / 365).cast("int")
)

display(pit_avg_by_finish_pos.head(5))

# the youngest and the oldest driver in each race, filter and concat

# find the minimum age
youngest = pit_avg_by_finish_pos.groupBy("raceId").agg(
    min("age").alias("min_age")
)

# find the maximum age
oldest = pit_avg_by_finish_pos.groupBy("raceId").agg(
    max("age").alias("max_age")
)

# match youngest age back to get driver name
youngest_drivers = pit_avg_by_finish_pos.join(youngest, on="raceId") \
    .filter(col("age") == col("min_age")) \
    .select("raceId", concat_ws(" ", "forename", "surname").alias("youngest_driver"), col("age").alias("youngest_age"))

# match oldest age back to get driver name
oldest_drivers = pit_avg_by_finish_pos.join(oldest, on="raceId") \
    .filter(col("age") == col("max_age")) \
    .select("raceId", concat_ws(" ", "forename", "surname").alias("oldest_driver"), col("age").alias("oldest_age"))

# combine the results into one table
result = youngest_drivers.join(oldest_drivers, on="raceId").orderBy("raceId")
display(result.head(5))


# ===========================
# 5. At any given race, how many podiums does each driver have?
# ===========================


# cast position, filter out non-finishers
res = race_results.filter(col("position") != "\\N") \
    .withColumn("position", col("position").cast("int"))

# cumulative podium count per driver up to each race
w = Window.partitionBy("driverId").orderBy("raceId").rowsBetween(Window.unboundedPreceding, Window.currentRow)

podiums = res.withColumn(
    "num_wins", sum(when(col("position") == 1, 1).otherwise(0)).over(w)
).withColumn(
    "num_2nd", sum(when(col("position") == 2, 1).otherwise(0)).over(w)
).withColumn(
    "num_3rd", sum(when(col("position") == 3, 1).otherwise(0)).over(w)
)

display(podiums.select("raceId", "driverId", "position", "num_wins", "num_2nd", "num_3rd").orderBy("driverId", "raceId").head(5))

podiums_slim = podiums.select("raceId", "driverId", "num_wins", "num_2nd", "num_3rd")

pit_avg_by_finish_pos = pit_avg_by_finish_pos.join(podiums_slim, on=["raceId", "driverId"], how="left")

display(pit_avg_by_finish_pos.head(5))


# ===========================
# 6. Does the average pit stop time improve (get shorter) over the years?
# ===========================


# extract year from race_date
pit_by_year = pit_avg_by_finish_pos.withColumn("race_year", year(to_date(col("race_date"))))

# average pit time per driver per year
yearly_avg = pit_by_year.groupBy("race_year", "driverId", "forename", "surname", "code").agg(
    round(avg("avg_pit_time"), 0).alias("yearly_avg_pit_time")
).orderBy("driverId", "race_year")

display(yearly_avg.head(5))

# It seems like the calculation for the duration of a pit stop have changed in year 2016,
# 2020-2024, since the time is significantly longer than the other years.
# 2021 is the slowest avg pit stop for all drivers

# investigate 2021 to find out what happened
pit_2021 = pit_by_year.filter(col("race_year") == 2021)
display(pit_2021.orderBy("race_date", "avg_pit_time").head(5))

# The year 2021 has a mix of seemingly normal pitstops and extremely long pitstop.
# My guess the accidents might contributed to the longer than average pitstops
