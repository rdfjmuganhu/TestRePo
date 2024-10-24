{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Final Assignment\n",
    "#### Mehmet Selim Çetin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Load Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy.stats\n",
    "import statsmodels.api as sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'\n",
    "boston_df=pd.read_csv(boston_url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>252.500000</td>\n",
       "      <td>3.613524</td>\n",
       "      <td>11.363636</td>\n",
       "      <td>11.136779</td>\n",
       "      <td>0.069170</td>\n",
       "      <td>0.554695</td>\n",
       "      <td>6.284634</td>\n",
       "      <td>68.574901</td>\n",
       "      <td>3.795043</td>\n",
       "      <td>9.549407</td>\n",
       "      <td>408.237154</td>\n",
       "      <td>18.455534</td>\n",
       "      <td>12.653063</td>\n",
       "      <td>22.532806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>146.213884</td>\n",
       "      <td>8.601545</td>\n",
       "      <td>23.322453</td>\n",
       "      <td>6.860353</td>\n",
       "      <td>0.253994</td>\n",
       "      <td>0.115878</td>\n",
       "      <td>0.702617</td>\n",
       "      <td>28.148861</td>\n",
       "      <td>2.105710</td>\n",
       "      <td>8.707259</td>\n",
       "      <td>168.537116</td>\n",
       "      <td>2.164946</td>\n",
       "      <td>7.141062</td>\n",
       "      <td>9.197104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.006320</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.460000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.385000</td>\n",
       "      <td>3.561000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>1.129600</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>187.000000</td>\n",
       "      <td>12.600000</td>\n",
       "      <td>1.730000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>126.250000</td>\n",
       "      <td>0.082045</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.190000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.449000</td>\n",
       "      <td>5.885500</td>\n",
       "      <td>45.025000</td>\n",
       "      <td>2.100175</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>279.000000</td>\n",
       "      <td>17.400000</td>\n",
       "      <td>6.950000</td>\n",
       "      <td>17.025000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>252.500000</td>\n",
       "      <td>0.256510</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.690000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.538000</td>\n",
       "      <td>6.208500</td>\n",
       "      <td>77.500000</td>\n",
       "      <td>3.207450</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>330.000000</td>\n",
       "      <td>19.050000</td>\n",
       "      <td>11.360000</td>\n",
       "      <td>21.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>378.750000</td>\n",
       "      <td>3.677083</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.624000</td>\n",
       "      <td>6.623500</td>\n",
       "      <td>94.075000</td>\n",
       "      <td>5.188425</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>666.000000</td>\n",
       "      <td>20.200000</td>\n",
       "      <td>16.955000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>505.000000</td>\n",
       "      <td>88.976200</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>27.740000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.871000</td>\n",
       "      <td>8.780000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>12.126500</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>711.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>37.970000</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Unnamed: 0        CRIM          ZN       INDUS        CHAS         NOX  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   \n",
       "mean   252.500000    3.613524   11.363636   11.136779    0.069170    0.554695   \n",
       "std    146.213884    8.601545   23.322453    6.860353    0.253994    0.115878   \n",
       "min      0.000000    0.006320    0.000000    0.460000    0.000000    0.385000   \n",
       "25%    126.250000    0.082045    0.000000    5.190000    0.000000    0.449000   \n",
       "50%    252.500000    0.256510    0.000000    9.690000    0.000000    0.538000   \n",
       "75%    378.750000    3.677083   12.500000   18.100000    0.000000    0.624000   \n",
       "max    505.000000   88.976200  100.000000   27.740000    1.000000    0.871000   \n",
       "\n",
       "               RM         AGE         DIS         RAD         TAX     PTRATIO  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   \n",
       "mean     6.284634   68.574901    3.795043    9.549407  408.237154   18.455534   \n",
       "std      0.702617   28.148861    2.105710    8.707259  168.537116    2.164946   \n",
       "min      3.561000    2.900000    1.129600    1.000000  187.000000   12.600000   \n",
       "25%      5.885500   45.025000    2.100175    4.000000  279.000000   17.400000   \n",
       "50%      6.208500   77.500000    3.207450    5.000000  330.000000   19.050000   \n",
       "75%      6.623500   94.075000    5.188425   24.000000  666.000000   20.200000   \n",
       "max      8.780000  100.000000   12.126500   24.000000  711.000000   22.000000   \n",
       "\n",
       "            LSTAT        MEDV  \n",
       "count  506.000000  506.000000  \n",
       "mean    12.653063   22.532806  \n",
       "std      7.141062    9.197104  \n",
       "min      1.730000    5.000000  \n",
       "25%      6.950000   17.025000  \n",
       "50%     11.360000   21.200000  \n",
       "75%     16.955000   25.000000  \n",
       "max     37.970000   50.000000  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "boston_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1.0</td>\n",
       "      <td>296.0</td>\n",
       "      <td>15.3</td>\n",
       "      <td>4.98</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>9.14</td>\n",
       "      <td>21.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>4.03</td>\n",
       "      <td>34.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>2.94</td>\n",
       "      <td>33.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>5.33</td>\n",
       "      <td>36.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0     CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  \\\n",
       "0           0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0   \n",
       "1           1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0   \n",
       "2           2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0   \n",
       "3           3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0   \n",
       "4           4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0   \n",
       "\n",
       "     TAX  PTRATIO  LSTAT  MEDV  \n",
       "0  296.0     15.3   4.98  24.0  \n",
       "1  242.0     17.8   9.14  21.6  \n",
       "2  242.0     17.8   4.03  34.7  \n",
       "3  222.0     18.7   2.94  33.4  \n",
       "4  222.0     18.7   5.33  36.2  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "boston_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Description of Dataset Variables\n",
    "The following describes the dataset variables:\n",
    "\n",
    "·      CRIM - per capita crime rate by town\n",
    "\n",
    "·      ZN - proportion of residential land zoned for lots over 25,000 sq.ft.\n",
    "\n",
    "·      INDUS - proportion of non-retail business acres per town.\n",
    "\n",
    "·      CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)\n",
    "\n",
    "·      NOX - nitric oxides concentration (parts per 10 million)\n",
    "\n",
    "·      RM - average number of rooms per dwelling\n",
    "\n",
    "·      AGE - proportion of owner-occupied units built prior to 1940\n",
    "\n",
    "·      DIS - weighted distances to five Boston employment centres\n",
    "\n",
    "·      RAD - index of accessibility to radial highways\n",
    "\n",
    "·      TAX - full-value property-tax rate per $10,000\n",
    "\n",
    "·      PTRATIO - pupil-teacher ratio by town\n",
    "\n",
    "·      LSTAT - % lower status of the population\n",
    "\n",
    "·      MEDV - Median value of owner-occupied homes in $1000's"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Generate Descriptive Statistics and Visualizations\n",
    "#### Median value of owner-occupied homes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Median value of owner-occupied homes')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot = sns.boxplot(x = 'MEDV', data = boston_df)\n",
    "plot.set_title('Median value of owner-occupied homes')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This plot shows that the distribution of MEDV is right skewed. Also there might be some outliers on the right side."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Charles River dummy variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Boundry to Charles River')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot2 = sns.countplot(x = 'CHAS', data = boston_df)\n",
    "plot2.set_title('Boundry to Charles River')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The histogram shows that the majority of the houses are not near the Charles River"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Boxplot for the MEDV variable vs the AGE variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "boston_df.loc[(boston_df['AGE'] <= 35), 'Age_Group'] = '<=35'\n",
    "boston_df.loc[(boston_df['AGE'] > 35) & (boston_df['AGE'] < 70), 'Age_Group'] = ' >35 and <70 '\n",
    "boston_df.loc[(boston_df['AGE'] >= 70), 'Age_Group'] = '>=70'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Median value of owner-occupied homes per Age Group')"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAbwAAAEWCAYAAAAdNyJXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAmU0lEQVR4nO3de5wcVZn/8c83IZhAkEuCqBlgxCQKKouaH7jIRlDJEgRv6AJ7cfCyrqswSFwV/SWYJWHF9cLusK6Kgg7I1VXUZZM18RIuooaES7gpE3SA4ZoJBEiIOCTP/lGnSU2ne6Zn0j2dSX3fr1e/pqvq1Kmnq6v6qXOqpkoRgZmZ2Y5uTLMDMDMzGwlOeGZmVghOeGZmVghOeGZmVghOeGZmVghOeGZmVghOeAUnKSRNTe+/Lmles2PKk/QdSQubsNx3SXpA0npJrx3p5e9oJH1W0reGOe98Sd+tMu1IST3bFp0VhRPeKCGpW9KfJE0uG39rSlqt27qMiPhIRCzY1np2EF8CTo2IiRFxS7ODGe0i4l8i4kPNjqMoJO2aDtYWNXg5knSqpFWSnpH0iKRlkk5q5HKHywlvdPkDcHJpQNJrgAnNC2eHtj9wZ7ODqIWknZodg428Qb739wDPArMkvaSBYXQAHwc+AUwCpgBzgWMqFU4Jsml5xwlvdLkEeF9uuA24OF9A0gskfUnS/ZIeTd2UE3LTPynpYUkPSfpA2bzPdx9K2lPSNZLWSHoivW/JlV0maYGkX0p6WtKS8tZnruzdko7LDe8kqVfS69Lw99KR4ZOSrpP0qir1nCLphrJx+S7ZAT972XxjJM2VdJ+kxyRdLGn3VMd6YCxwm6R7q8x/uKSbUsw3STo8jT9K0u25cj+VtDw3fIOkd6b33ZL+KR0dPynpSknjc2WPSy34dZJulHRwblq3pE9LWgVsqPTjVy3GNG0vSd9O28ETkn6Ym/aOtNynJN0r6ZjcMt+aK/d8V6Ok1vRdfDjV+bCkT1Qqm4bfkD7TOkm3SToyN+1lkq5N29VSoOJ2VfZZP5G+x4clvT83fvf03a5J3/VcpR/ctD39UtJ5KY7fp3V2irLu7MckteXqqrp9SZqsbB9ZJ+lxSderyg97Wk/taXm9kr6YLyvpA8r2mSck/UTS/mXzfkxSF9A1wCppA74OrAL+pmz5r5N0S1q/30vb3cLc9KrbXVk904GPAidFxNKI2BgRmyLihog4JVdumaRzJP0SeAY4YJBtc9jb2aAiwq9R8AK6gbcCvwMOJPtBfoCsJRJAayr3b8CPgb2A3YD/Bj6fph0DPAq8GtgVuCzNOzVN/w6wML2fBJwA7JLq+R7ww1w8y4B7gelkrcxlwLlVYj8LuDQ3/Dbgt7nhD6RlvCDFf2tuWj6mU4AbyurOx1/1s1eI6QPAauAAYCLwA+CSSvVWmHcv4Ang74CdyFrdT6R1Nh7YSPYjvRPwCPBQimdCmjYp950uB16a6rwb+Eia9jrgMeCw9F23pfIvyM17K7AvMGEoMabp/wNcCewJjAPelMYfCjwJHE12QDwFeGV+G8wtYz7w3fS+Na2zy8m2rdcAa0rly8pOAdYCx6ZlHJ2G907TfwV8JW0PM4GnS/NW+JxHAs8BZ6fPcSzZj+qeafrFwI/S+m8F7gE+mNuengPen9bxQuB+4Ktp2bPSsifWsG99nizBjEuvvwBUJeYAfpHq2S/F9KE07Z1k2+WB6XubC9xYNu/SNO9W33sqsx+wGTiIrOW1KjdtZ+A+4PQU57uBP7FlHxtwuytbzkeA7hp+u5al9fqq9Jn2YeBts5thbmeDxjKSP9p+Df/FloQ3N+1cx6QNf6e0AbQCAjYAL8/N9+fAH9L7i8glJbJkVTHhVVj+IcATueFlwNzc8EeB/60y71SyH45d0vClwFlVyu6RYtq9PCYGSHiDffYKy/kZ8NHc8CuAPmCnfL1V5v07YHnZuF8Bp6T315P9kLwBWAJclb6vo+j/49MN/G1u+F+Br6f3XwMWlC3jd2xJTN3ABwbYXqrGCLyE7AdxzwrzfQM4b6BtMDc8n61/iF5Z9nkurFD20+QOLtK4n5D9uO5HloR2zU27jIET3sbS95bGPZbW/Viybr2DctP+AViW2566ctNekz7DPrlxa8m2/cH2rbPJEmvFbabCNntM2b7zs/R+MSkhp+ExZAl8/9y8bx6k/rmkg0ayg6lNwGvT8EzgQXLJGLiBLfvYgNtdheX8umxcD7AO+GMu5mXA2UPYf4a9nQ32ct//6HMJcB3wMsq6M4G9yVpkKyWVxolsx4ds41+ZK39ftYVI2gU4j+yHes80ejdJYyNiUxp+JDfLM2Qtpa1ExGpJdwPHS/pv4O3Aa9NyxgLnAO9N8W9Os00ma2nUarDPXu6l9P/897Hl6PPBQZZVPm9p/inp/bVkP8Q96f0TwJvIfnyvLZuvfB2+NL3fH2iTdFpu+s656ZC18JG0H3BXaWRETBwkxn2BxyPiiQqfbV9gWy50eKBsea+pUGZ/4L2Sjs+NG0fW6nkp2YHVhrJ69h1gmWsj4rnccGlbnMyWFk2+rim54Udz7zcCRET5uIkMvn19keyHeUmafkFEnDtAzOXrKf+9/7ukL+emK8V8X4V5K3kf8M30WR6SdC3ZwcQtaTkPRsoUFeqrZbsrWUt28PS8iGhR1r3el+KutIzB9p9a1LKdbcXn8EaZiLiP7OKVY8m64fJ6yXbQV0XEHum1e/oBBHiY/j8c+w2wqE+QtXoOi4gXkh0ZQv+NeCguJ+u6eAdwV0SsTuP/Oo17K7A72RFcteVsIPvRyQpIL85NG+yzl3uIbOcuKbUsHq1cfMB5S/OXEmUp4c1M768lS3hvYuuEV80DwDm5z7JHROwSEZfnymSH/BH3R3Y16cTc5x0oxgeAvSTtUWW5L68SU7/1D7y4Qpny7euhKsu4pOyz7ZoSxMPAnpJ2LatnOHrJfnjLv+fBDmiq1VV1+4qIpyPiExFxAHA8MEfSWwaor9p6egD4h7J1MyEibsyVzyerftK5sGnAZ5SdF3+ErHvy5JSIHgamKJe1y2KpZbsr+TnQImnGAJ+zUsyD7T/12s624oQ3On2QrFsjfxRMRGwmO7I7T9KLACRNkfSXqchVwCmSDkotuM8NsIzdyHbwdZL2GqRsLa4gOyfyj2RdVPnlPEt2tLgL8C8D1HEb8CpJhyi7uGN+aUINn73c5cAZ6QKJiWm5V5a1FKpZBEyX9NfKLsA5kex8yTVp+o1kBwuHknXd3Em2gx9G1jqvxTeBj0g6TJldJb1N0m41zl81xoh4mKzr7D+VXZw0TlLpgOZC4P2S3qLswp4pkl6Zpt0KnJTKzyC7ErDcPEm7KLvw6P1k5wnLfZestf+XksZKGq/s/+la0gHdCuCfJe0s6QiyBDJkqSfiKuAcSbuliz/mpOUPta4Bt690ocfUlEieIutG3FS1QvhkWvf7kp1PK62nr5Mlq1eleneX9N4hhNpGdqrjILKu2EPIztnvAswm6zrcBJyatot3kG2nJTVvdxHxO7Iu8CskHS1pQuqxOby8bJnB9p9bqc92thUnvFEoIu6NiBVVJn+a7KT3ryU9BfyU7MeXiFhMduL956nMzwdYzL+RXWTRC/wa+N9tjPlhsp3tcPpvnBeTdUk8SNYt9+sB6riH7FzJT8muULuhrEjVz17BRWzpHv4D2TmH06qULY9jLXAcWSt4LfAp4LiI6E3TNwA3A3dGxJ/SbL8C7ouIx2pcxgrg74H/IOsSXU12zqkmg8VIdh6lD/gt2Tmvj6f5lpP9gJxH1qV8LVuOxueRtf6eAP6Z/gcuJdemWH8GfCkillSI7QGyVv1nyS44eAD4JFt+j/6a7ODgcbIDrfKu+6E4jazF8Huy7eUysu9+OAbavqal4fVk3/V/RsSyAer6EdnphVvJLiC6ECAirga+QJZEngLuIEtUg0oHgX8FnB8Rj+RefyDb1tvS9vhusoPmdcDfkiWaZ9Pyh7rdfYzsXxO+QvZ99QALgBPJLlTZSg3bZl22s0rUvyvXzGzolN344A/AuBpbyYUlKYBpuW79ppL0G7KLpb7d7FgGs63bmVt4ZmYFIulNkl6cuhPbgIPZxh6c0cJXaZqZFcsryM5tTiT7X9r3pFMOOzx3aZqZWSG4S9PMzArBXZrbqcmTJ0dra2uzwzAzG1VWrlzZGxF7V5rmhLedam1tZcWKav95YGZmlUiqegcpd2mamVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkh+CpNsybp6Ohg9eqh3U6xp6cHgJaWlkaEVJOpU6fS3t7etOWbDZcTnlmTrF69mltuv4vNu+xV8zxjnsmeifvos83Zdcc883hTlmtWD054Zk20eZe9+ONBx9Vcfvxd2SPDhjJPPZWWbzYa+RyemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOemZkVghOeFVpHRwcdHR3NDsNGGW83o5NvHm2FNtTH85iBt5vRyi08MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrhO0m4UlaIGmVpFslLZH00jS+VdLGNP5WSV9v0PKXSZoxjPnOy8V2j6R1uWltkrrSq62uAY8Svb29nHbaaaxdu7bZoZgVUmkf7OrqqnlfLN9vR3I/vueee5g9e3ZDbt82YglP0p6DFPliRBwcEYcA1wBn5abdGxGHpNdHGhbkEJQ+T0ScUYoNOB/4QZq+F/A54DDgUOBzNayDHU5nZyerVq2is7Oz2aGYFVJpH1ywYEHN+2L5fjuS+/HChQvZsGEDZ599dt3rHskW3g8l/VjS2yVtddPqiHgqN7grEEOpXNJZkm6SdIekCyQpjV8m6QuSlqcW2F+k8RMkXZFalVcCE2pYxosk/ZOkO4ATKxQ5Gbg8vf9LYGlEPB4RTwBLgWOG8plGu97eXhYvXkxEsHjxYrfyzEZYfh/s7u6uaV8s32+7urpGbD++55576O7uBqC7u7vurbyRfFrCkcBM4APAVyR9D7gwIp7/RJLOAd4HPAkclZv3ZZJuAZ4C5kbE9RXq/4+IODvVcwlwHPDfadpOEXGopGPJWl1vBf4ReCYiDpZ0MHBzpaAljQFmAR8CDgIuA46JiJ6ycvsDLwN+nkZNAR7IFelJ4wqjs7OTiOy4ZfPmzXR2djJnzpwmR9VfT08PGzdupL29fcSX3dXVhf40pOO6ptMfn6Kr6+mmrK/tSVdXFxMmDHqM3HT5fbBksH2xfL9dsGDBiO3HCxcu7Dd89tlnc/HFF9et/hFr4UXm2ohoA14HbAZ+K+mEXJn/HxH7ApcCp6bRDwP7RcRrgTnAZZJeWGERR0n6jaTbgTcDr8pN+0H6uxJoTe9nAt9Ny10FrKoS+g+Bb6XXqyJiYXmyS04C/isiNqVhVVoNVZaRzSB9WNIKSSvWrFkzUNFRYenSpfT19QHQ19fHkiVLmhyRWbHk98GSwfbF8v22u7t7xPbjUuuu2vC2GtHn4UmaALyLrJW3B3A6WVdfucuA/wE+FxHPAs8CRMRKSfcC04EVuXrHA/8JzIiIByTNB8bn6ns2/d1E/89cy+H1Z4C/Jzs/t1TStyPipgrlTgI+lhvuIWvVlrQAywZaUERcAFwAMGPGjNF16F/B0UcfzaJFi+jr62PcuHHMmjWr2SFtpaWlBaApD/Nsb29n5b2PjPhyt0WMfyHTXv7iwj/8dLS0cPP7YMlg+2L5fjtlyhQefPDBEdmPW1tb+yW51tbWutY/khet/CtwF/BG4JMRMSMivlo6dydpWq7424HfpvF7Sxqb3h8ATAN+X1Z9Kbn1SpoIvKeGkK4D/ibV+2rg4EqFIuLOiPg4WYvxWuCcdN7v+W9d0iuAPYFf5Wb9CTBL0p7pYpVZaVxhtLW1kU6lMmbMGNraCnmhqlnT5PfBksH2xfL9dt68eSO2H8+dO7ff8FlnnVWl5PCM5EUry4ADI+JjEXFLhennpgtOVpElh9PT+JnAKkm3Af8FfCQiHs/PGBHrgG8Ct5N1QVZqgZX7GjAxLe9TwPKBCkfEnyLiyoiYBRwP5M/cngxcEbnO8hTjghTLTcDZ5XHv6CZPnszs2bORxOzZs5k0aVKzQzIrlPw+2NraWtO+WL7fTps2bcT24+nTpz/fqmttbWXq1Kl1rX/EujQjYtEg00+oMv77wPdrqH8uMLfC+CNz73tJ5/AiYiNZN+SQRcR9wH254flVyl0EXDScZewo2tra6O7uduvOrElK+2B7ezsdHR017Yvl++1I7sdz587l9NNPr3vrDkb4HJ4Vz+TJkzn//PObHYZZYeX3wVr3xfL9diT34+nTp7N48eKG1L3d3GnFzMyskZzwzMysEJzwzMysEJzwzMysEJzwzMysEJzwzMysEJzwzMysEJzwzMysEPyP51Zo9b51kRWDt5vRyQnPCm203PXeti/ebkYnd2mamVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkh+ObRZk005pnHGX/XNUMovxZgSPPU05hnHgde3JRlm20rJzyzJhnOI2Z6ep4DoKWlWUnnxX40jo1aTnhmTeJHzJiNLJ/DMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQnDCMzOzQvC9NG1U6+joYPXq1SO+3J6eHgBaWlpGdLlTp071PTjNhskJz0a11atXc88dN7PfxE0jutwNT48F4I/PPTxiy7x//dgRW5bZjsgJz0a9/SZuYu6M9SO6zIUrJgKM6HJLyzSz4fE5PDMzKwQnPDMzK4QhJTxJL5S0W6OCMTMza5SaEp6kGZJuB1YBd0i6TdLrGxuamZlZ/dR60cpFwEcj4noASUcA3wYOblRgZmZm9VRrl+bTpWQHEBE3AE83JiQzM7P6q7WFt1zSN4DLgQBOBJZJeh1ARNzcoPjMzMzqotaEd0j6+7my8YeTJcA31ysgMzOzRqgp4UXEUY0OxMzMrJFqSniSzqo0PiLOrm84ZmZmjVFrl+aG3PvxwHHA3fUPx8zMrDFq7dL8cn5Y0peAHzckIjMzswYY7q3FdgEOqGcgVj8dHR10dHQ0OwyzpvJ+YOVqPYd3O9nVmABjgb0Bn7/bTjXj+XBm2xvvB1au1nN4x+XePwc8GhHPNSAeMzOzhqipSzMi7gP2AI4H3gUc1MCYzMzM6q7Wm0efDlwKvCi9LpV0WiMDMzMzq6dauzQ/CBwWERsAJH0B+BVwfqMCMzMzq6dar9IUsCk3vCmNMzMzGxWG8nig30i6Og2/E7iwIRGZmZk1wKAJT9IY4DfAtcARZC2790fELQ2OzczMrG4GTXgRsVnSlyPizwE/BsjMzEalWs/hLZF0giSftzMzs1Gp1oQ3B/ge8KykpyQ9LempBsZlZtY0vb29nHbaaaxdu7bicKWyXV1dFcvUMm+ladXK1TJPrfUOt/xQDSX+M888k5kzZzJ37ty6x1HrP57vFhFjImLniHhhGn5h3aMZ5SR9UtKt6XWHpE2S9krTjpH0O0mrJZ3Z7FjNrLrOzk5WrVpFZ2dnxeFKZRcsWFCxTC3zVppWrVwt89Ra73DLD9VQ4r/xxhsBuO666+oex4AJT9JYSRNzw2+QNDO9dqt7NE0mac9tmT8ivhgRh0TEIcBngGsj4nFJY4GvArPJ7lJzsiTfrcZsO9Tb28vixYuJCBYvXkxXV1e/4XzLJF+2u7t7qzLldVWbt3xatXgWLVo06Dy11jvc8kM1lPjPPLN/W6DerbzBLlr5AvAY8K9p+HLgDrJn4t0MfLqu0TTfCkm/Ab4F/CIiYrAZBnAy2foCOBRYHRG/B5B0BfAO4K5tCbaanp4eNm7cSHt7eyOq3650dXWxc99wH/oxujz6zBj+1NVViO+1Hrq6upgwYcKQ5+vs7KS062/evJkFCxb0G+7s7GTOnDlblS3Jlymvq9q85dOqxdPX11dxOQPFX63e4ZYfqqHEX2rdldS7lTfYL8VbgK/khtdFxPHALOCNdY1k+zAduAw4FbhL0mclvbQ0UdJ5uS7L/KvfYYmkXYBjgO+nUVOAB3JFetI4yub7sKQVklasWbOmzh/NzGqxdOnS53+Y+/r66O7u7je8ZMmSimVL8mXK66o2b/m0avFERL/kUWmeWusdbvmhGmr8jTRYC29M2VMRPg0QEZHv6txRRMQm4BrgGkl7A58H7pd0eEQsj4gzaqzqeOCXEfF4Gq50detWrceIuAC4AGDGjBnDbl22tLQAFOJZYO3t7fyx+6ZmhzEi9tllM+NbpxXie62H4baEjz76aBYtWkRfXx/jxo1jypQpPPjgg88Pz5o1q2LZknyZ8rqqzVs+rVo8pQvlI6LqPLXWO9zyQzXU+BtpsBbezvlzdRGxBEDS7mTdmjscSbtL+jDZE92nk91HdFWaVlMLDziJLd2ZkLXo9s0NtwAPNe5TmNlwtbW1Pf/DPGbMGObNm9dvuK2trWLZknyZ8rqqzVs+rVo848aNY9y4cQPOU2u9wy0/VEOJ//DDD+83PHPmzLrGMljC+yZwpaT9SiMk7U/2Y/7NukayHZD0XbJzkwcA74uImRHRGRF/BIiIM0oXpZS9zs3VsTvwJuBHuapvAqZJepmknckS4o9H7IOZWc0mT57M7NmzkcTs2bOZNm1av+FJkyZVLNva2rpVmfK6qs1bPq1aPMcee+yg89Ra73DLD9VQ4j/33HP7DS9cuLCusQzYpRkRX5H0DHCDpF3JuuE2AOdGxNfqGsn24SrglG18uO27gCWlJ0sARMRzkk4FfkL2xPiLIuLObQvVzBqlra2N7u7ufi21/HClsu3t7XR0dGxVppZ5a2mFlcqVrggdaJ5a6x1u+aEaSvyHH344N954Y91bdwCq9ULEdM5OEfF0hWltEdGYf+AoqBkzZsSKFSuGNW/p3EURzvWUzuHNnbF+RJe7cEV2Cnskl7twxUTGt/6/Qnyv9VCk/cC2kLQyImZUmlbz9dwRsb5SsktOH1ZkZmZmI6Re/8Dke2yamdl2rV4Jb1v+QdvMzKzh3MIzM7NCqFfC+2Wd6jEzM2uImhKepH0kXShpcRo+SNIHS9Mj4tRGBWhmZlYPtbbwvkP2P2Sl+0reA3y8AfGYmZk1RK0Jb3JEXAVshuwfqYFNDYvKzMyszmpNeBskTSJdjSnpDcCTDYvKzMyszgZ7WkLJHLJ7P75c0i+BvYH3NCwq2yZTp05tdghmTef9wMrVlPAi4mZJbwJeQfYvCL+LiL5BZrMm8QNCzbwf2NZqSniS3l02arqkJ4HbI+Kx+odlZmZWX7V2aX4Q+HPgF2n4SODXZInv7Ii4pAGxmZmZ1U2tCW8zcGBEPArZ/+UBXwMOA64DnPDMzGy7VutVmq2lZJc8BkyPiMcBn8szM7PtXq0tvOslXQN8Lw2fAFyXHgq7rhGBmZmZ1VOtCe9jwLuBI9LwcuAl6aneRzUiMDMzs3qqqUszssei30vWffku4C3A3Q2My8zMrK4GbOFJmg6cBJwMrAWuBBQRbtWZmdmoMliX5m+B64HjI2I1gKQzGh6VmZlZnQ3WpXkC8AjwC0nflPQW/LBXMzMbhQZMeBFxdUScCLwSWAacAewj6WuSZo1AfGZmZnVR6700NwCXApdK2gt4L3AmsKSBsZnV5P71Y1m4YuKILvO+p8cCjOhy718/lukjtjSzHU+t/5bwvPTP5t9IL7OmatYd8Xft6QFgfEvLiC1zOn4CgNm2GHLCM9ue+I74ZlarWm8tZmZmNqo54ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSH45tFmo1RHRwerV6+uuXxPesJDywg+4WFbTJ061TcHt7pywjMbpVavXs0td94Ce9Q4w5PZnzVa06iQ6mddswOwHZETntlotgdsPnJzTUXHLMvOYNRavplKsZrVk7cqMzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8MzMrBCc8syo6Ojro6OhodhhmNfM2OzDfPNqsiqE8esdse+BtdmBu4ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44ZmZWSE44dWRpHdIWiXpVkkrJB2Rm9Yt6fbStGbGaWbF1Nvby2mnncbatWtHvK5ayy9fvpwjjzySlStXbnOM5ZzwBiFpZ0m71lj8Z8CfRcQhwAeAb5VNPyoiDomIGfWM0cysFp2dnaxatYrOzs4Rr6vW8vPnz2fz5s3Mmzdvm2Ms54RXhaQDJX0Z+B0wvZZ5ImJ9REQa3BWIgcqbmY2U3t5eFi9eTESwePHibWrlDbWuWssvX76c9evXA7B+/fq6t/L8tISc1JL7K+CDgIBvAwdHxNNp+nnAURVmvSIizk1l3gV8HngR8LZcmQCWSArgGxFxQcM+iNVFT08PGzdupL29vdmhVNTV1QWbmx1Fg6zPPt/2uu63V11dXUyYMKHitM7OTkrH45s3b6azs5M5c+YMazlDravW8vPnz+83PG/ePBYtWjSsGCtxC6+/h8mS3Yci4o0R8a1SsgOIiDNSl2T569xcmasj4pXAO4EFubrfGBGvA2YDH5M0s3zhkj6czv2tWLNmTaM+o5kV0NKlS+nr6wOgr6+PJUuWjFhdtZYvte6qDW8rt/D6ew9Zwrta0uVAZ0TcV5pYSwuvJCKuk/RySZMjojciHkrjH5N0NXAocF3ZPBcAFwDMmDHD3aFN1tLSArDdPlCzvb2dWx68pdlhNMZEmDZl2na77rdXA7WIjz76aBYtWkRfXx/jxo1j1qxZw17OUOuqtfzEiRP7JbmJEycOO8ZK3MLLiYglEXEicATwJPAjST+V1JqmD9jCkzRVktL71wE7A2sl7SpptzR+V2AWcMfIf0IzK6q2tjbSzxNjxoyhra1txOqqtXx5l+aCBQsqlhsuJ7wKImJtRPx7utrys8CmGmc9AbhD0q3AV4ET00Us+wA3SLoNWA78T0T8b/0jNzOrbPLkycyePRtJzJ49m0mTJo1YXbWWP/TQQ59v1U2cOJHXv/71w46xEndpDiIilg+h7BeAL1QY/3vgz+oZl5nZULW1tdHd3b1Nrbvh1lVr+fnz5/OpT32q7q07cMIzMyuMyZMnc/755zelrlrLH3rooSxbtmwbIqvOXZpmZlYITnhmZlYITnhmZlYITnhmZlYITnhmZlYITnhmZlYITnhmZlYITnhmZlYI/sdzsyqmTp3a7BDMhsTb7MCc8Myq8LPYbLTxNjswd2mamVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkhOOGZmVkh+ObRZqPZOhizrMbj1nXZn5rLN9M6YEqzg7AdjROe2Sg11EfB9EQPAC1TWhoRTn1N8aNurP6c8MxGKT8KxmxoRkHfhpmZ2bZzwjMzs0JwwjMzs0JwwjMzs0JwwjMzs0JQRDQ7BqtA0hrgvmbHUQeTgd5mB7Gd8Lroz+tjC6+L/rZlfewfEXtXmuCEZw0laUVEzGh2HNsDr4v+vD628Lror1Hrw12aZmZWCE54ZmZWCE541mgXNDuA7YjXRX9eH1t4XfTXkPXhc3hmZlYIbuGZmVkhOOGZmVkhOOFZXUi6SNJjku7IjdtL0lJJXenvns2McSRJ2lfSLyTdLelOSaen8YVbJ5LGS1ou6ba0Lv45jS/cuiiRNFbSLZKuScOFXRcAkrol3S7pVkkr0ri6rxMnPKuX7wDHlI07E/hZREwDfpaGi+I54BMRcSDwBuBjkg6imOvkWeDNEfFnwCHAMZLeQDHXRcnpwN254SKvi5KjIuKQ3P/f1X2dOOFZXUTEdcDjZaPfAXSm953AO0cypmaKiIcj4ub0/mmyH7cpFHCdRGZ9GhyXXkEB1wWApBbgbcC3cqMLuS4GUfd14oRnjbRPRDwMWQIAXtTkeJpCUivwWuA3FHSdpC68W4HHgKURUdh1Afwb8Clgc25cUddFSQBLJK2U9OE0ru7rxE88N2sgSROB7wMfj4inJDU7pKaIiE3AIZL2AK6W9Oomh9QUko4DHouIlZKObHI425M3RsRDkl4ELJX020YsxC08a6RHJb0EIP19rMnxjChJ48iS3aUR8YM0utDrJCLWAcvIzvcWcV28EXi7pG7gCuDNkr5LMdfF8yLiofT3MeBq4FAasE6c8KyRfgy0pfdtwI+aGMuIUtaUuxC4OyK+kptUuHUiae/UskPSBOCtwG8p4LqIiM9EREtEtAInAT+PiL+lgOuiRNKuknYrvQdmAXfQgHXiO61YXUi6HDiS7LEejwKfA34IXAXsB9wPvDciyi9s2SFJOgK4HridLedqPkt2Hq9Q60TSwWQXHYwlO8i+KiLOljSJgq2LvNSl+U8RcVyR14WkA8hadZCdZrssIs5pxDpxwjMzs0Jwl6aZmRWCE56ZmRWCE56ZmRWCE56ZmRWCE56ZmRWCE55ZgUkKSZfkhneStCZ3F/9T0vCtuddBklolbUx3/L87PQ2hLc1zpKRflS1nJ0nP/yOxWTP41mJmxbYBeLWkCRGxETgaeLCszJURcWp+RLo/6L0R8do0fADwA0ljyP7nrkVSa0R0p1neCtxRujeiWTO4hWdmi8nu3g9wMnD5UCuIiN8Dc4D2iNgMfA84MVfkpOHUa1ZPTnhmdgVwkqTxwMFkd4PJO7GsS3NClXpuBl6Z3l9OluSQ9ALgWLL7ipo1jbs0zQouIlalLsqTgUUVilTq0qxU1fMjI+ImSRMlvQI4EPh1RDxRv6jNhs4Jz8wgu1Hvl8juhzppmHW8lv5P8b6CrJV3IO7OtO2AE56ZAVwEPBkRtw/nOW2phfgl4Pzc6MvJ7nC/O/DBbQ/RbNs44ZkZEdED/HuVySempz+UfBR4CHi5pFuA8cDTwPkR8e1cnXdJegZYGREbGhS6Wc38tAQzMysEX6VpZmaF4IRnZmaF4IRnZmaF4IRnZmaF4IRnZmaF4IRnZmaF4IRnZmaF8H9OJh5/dM2pHwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot3 = sns.boxplot(x = 'MEDV', y = 'Age_Group', data = boston_df)\n",
    "plot3.set_title('Median value of owner-occupied homes per Age Group')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The boxplot above shows that on average the median value of owner occupied homes is higher when the Age is lower. But there are some outliers as well."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Nitric oxide concentration per proportion of non-retail business acres per town')"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot4 = sns.scatterplot(y = 'NOX', x = 'INDUS', data = boston_df)\n",
    "plot4.set_title('Nitric oxide concentration per proportion of non-retail business acres per town')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Values in the bottom-left section of the scatter plot indicates a strong relation between low Nitric oxide concentration and low proportion of non-retail business acres per town. \n",
    "\n",
    "Generally, a higher proprtion of non-retail business acres per town produces a higher concentration of Nitric oxide."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Pupil to teacher ratio "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Pupil to teacher ratio per town')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot5 = sns.countplot(x = 'PTRATIO', data = boston_df)\n",
    "plot5.set_title('Pupil to teacher ratio per town')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tests for Statistics\n",
    "#### Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hypothesis:\n",
    "\n",
    "H_0: There's no significant difference in median value between houses bounded and not bounded by the Charles River\n",
    "\n",
    "H_a: There's a significant difference in median value between houses bounded and not bounded by the Charles River"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "      <th>Age_Group</th>\n",
       "      <th>CHAS_T</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1.0</td>\n",
       "      <td>296.0</td>\n",
       "      <td>15.3</td>\n",
       "      <td>4.98</td>\n",
       "      <td>24.0</td>\n",
       "      <td>&gt;35 and &lt;70</td>\n",
       "      <td>FAR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>9.14</td>\n",
       "      <td>21.6</td>\n",
       "      <td>&gt;=70</td>\n",
       "      <td>FAR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2.0</td>\n",
       "      <td>242.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>4.03</td>\n",
       "      <td>34.7</td>\n",
       "      <td>&gt;35 and &lt;70</td>\n",
       "      <td>FAR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>2.94</td>\n",
       "      <td>33.4</td>\n",
       "      <td>&gt;35 and &lt;70</td>\n",
       "      <td>FAR</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3.0</td>\n",
       "      <td>222.0</td>\n",
       "      <td>18.7</td>\n",
       "      <td>5.33</td>\n",
       "      <td>36.2</td>\n",
       "      <td>&gt;35 and &lt;70</td>\n",
       "      <td>FAR</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0     CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  \\\n",
       "0           0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0   \n",
       "1           1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0   \n",
       "2           2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0   \n",
       "3           3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0   \n",
       "4           4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0   \n",
       "\n",
       "     TAX  PTRATIO  LSTAT  MEDV      Age_Group CHAS_T  \n",
       "0  296.0     15.3   4.98  24.0   >35 and <70     FAR  \n",
       "1  242.0     17.8   9.14  21.6           >=70    FAR  \n",
       "2  242.0     17.8   4.03  34.7   >35 and <70     FAR  \n",
       "3  222.0     18.7   2.94  33.4   >35 and <70     FAR  \n",
       "4  222.0     18.7   5.33  36.2   >35 and <70     FAR  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "boston_df.loc[(boston_df['CHAS'] == 0), 'CHAS_T'] = 'FAR'\n",
    "boston_df.loc[(boston_df['CHAS'] == 1), 'CHAS_T'] = 'NEAR'\n",
    "boston_df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ttest_indResult(statistic=-3.996437466090509, pvalue=7.390623170519905e-05)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scipy.stats.ttest_ind(boston_df[boston_df['CHAS_T'] == 'FAR']['MEDV'], \n",
    "                      boston_df[boston_df['CHAS_T'] == 'NEAR']['MEDV'], equal_var = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The p-value is less than 0.05, we reject the Null Hypothesis, which means that there is not a statistical difference in median value betwenn houses near the Charles River and houses far away."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hypothesis\n",
    "    \n",
    "H_0: There isn't statistical difference in Median values of houses (MEDV) for each proportion of owner occpied units built prior to 1940\n",
    "    \n",
    "H_a: There is statistical difference in Median values of houses (MEDV) for each proportion of owner occpied units built prior to 1940\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             df        sum_sq      mean_sq          F        PR(>F)\n",
      "AGE         1.0   6069.761065  6069.761065  83.477459  1.569982e-18\n",
      "Residual  504.0  36646.534350    72.711378        NaN           NaN\n"
     ]
    }
   ],
   "source": [
    "from statsmodels.formula.api import ols\n",
    "lm = ols('MEDV ~ AGE', data = boston_df).fit()\n",
    "table = sm.stats.anova_lm(lm)\n",
    "print(table)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Given p-value is less than 0.05, we fail to accept the Null Hypothesis --> There is statistical difference in Median values of houses (MEDV) for each proportion of owner occpied units built prior to 1940"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Hypothesis\n",
    "\n",
    "H_0: Nitric Oxide concentration is not correlated with the proportion of non-retail business acres per town\n",
    "\n",
    "H_a: Nitric Oxide concentration is correlated with the proportion of non-retail business acres per town"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.7636514469209151, 7.913361061238693e-98)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Given the Pearson Coefficient is 0.76365 and p-value less than 0.05, we reject the Null Hypothesis as there is a positive correlation between Nitric oxide concentration and proportion of non-retail business acres per town\n",
    "\n",
    "The positive relationship is confirmed also with the Scatter Plot (Question 4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### What is the impact of an additional weighted distance  to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramFiles\\lib\\site-packages\\statsmodels\\tsa\\tsatools.py:142: FutureWarning: In a future version of pandas all arguments of concat except for the argument 'objs' will be keyword-only\n",
      "  x = pd.concat(x[::order], 1)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th> <td>   0.062</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.061</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   33.58</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Sun, 14 May 2023</td> <th>  Prob (F-statistic):</th> <td>1.21e-08</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>03:26:20</td>     <th>  Log-Likelihood:    </th> <td> -1823.9</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   3652.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>   504</td>      <th>  BIC:               </th> <td>   3660.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th> <td>   18.3901</td> <td>    0.817</td> <td>   22.499</td> <td> 0.000</td> <td>   16.784</td> <td>   19.996</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>DIS</th>   <td>    1.0916</td> <td>    0.188</td> <td>    5.795</td> <td> 0.000</td> <td>    0.722</td> <td>    1.462</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>139.779</td> <th>  Durbin-Watson:     </th> <td>   0.570</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 305.104</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 1.466</td>  <th>  Prob(JB):          </th> <td>5.59e-67</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 5.424</td>  <th>  Cond. No.          </th> <td>    9.32</td>\n",
       "</tr>\n",
       "</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:                   MEDV   R-squared:                       0.062\n",
       "Model:                            OLS   Adj. R-squared:                  0.061\n",
       "Method:                 Least Squares   F-statistic:                     33.58\n",
       "Date:                Sun, 14 May 2023   Prob (F-statistic):           1.21e-08\n",
       "Time:                        03:26:20   Log-Likelihood:                -1823.9\n",
       "No. Observations:                 506   AIC:                             3652.\n",
       "Df Residuals:                     504   BIC:                             3660.\n",
       "Df Model:                           1                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const         18.3901      0.817     22.499      0.000      16.784      19.996\n",
       "DIS            1.0916      0.188      5.795      0.000       0.722       1.462\n",
       "==============================================================================\n",
       "Omnibus:                      139.779   Durbin-Watson:                   0.570\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):              305.104\n",
       "Skew:                           1.466   Prob(JB):                     5.59e-67\n",
       "Kurtosis:                       5.424   Cond. No.                         9.32\n",
       "==============================================================================\n",
       "\n",
       "Notes:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = boston_df['DIS']\n",
    "y = boston_df['MEDV']\n",
    "\n",
    "x = sm.add_constant(x)\n",
    "\n",
    "model = sm.OLS(y, x).fit()\n",
    "predisction = model.predict(x)\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The coef DIS of 1.0916 indicates that an additional weighted distance to the 5 empolyment centers in boston increases of 1.0916 the median value of owner occupied homes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
#Display the outputs 
print("New Python file") 
