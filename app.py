import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


import mymodule

#arr = mymodule.make_data()

# 1 made empty repository on github
# 2 PyCharm Project from github: .py script that is github, made script & requirements.txt, commit & pushed
# 3 Log into streamlit, and app is there...

# PRIORS - > USER INPUT
st.header('When you enter the geothermal lottery without further information?')
st.subheader('What\'s the Prior Probability $Pr(.)$ of a POSITIVE geothermal site?')
Pr_prior_POS_demo = np.linspace(0.05,1,20) #mymodule.Prior_probability_binary()

#### start of paste  -> CHANGE to input
count_ij = np.zeros((2,10))
value_array = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
st.write('value_array', value_array)

## Calculate Vprior
#f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
value_drill_DRYHOLE = np.linspace(100, -1e6,10)

# This function can be called with multiple values of "dry hole"
vprior_INPUT_demo = mymodule.f_VPRIOR([1-Pr_prior_POS_demo[0],Pr_prior_POS_demo[0]], value_array, value_drill_DRYHOLE[-1])       
# l2 = list(map(lambda v: v ** 2, l1))
vprior_INPUT_demo_list = list(map(lambda vv: mymodule.f_VPRIOR([1-Pr_prior_POS_demo[0],Pr_prior_POS_demo[0]], 
                                                              value_array,vv),value_drill_DRYHOLE))
st.subheader('Yes if Vprior is positive. Vprior with $Pr(POSITIVE)$='+str(Pr_prior_POS_demo[0]))

firstfig = plt.figure()
plt.plot(value_drill_DRYHOLE, vprior_INPUT_demo_list,'ks')
plt.ylabel('$V_{prior}$')
plt.xlabel('Dryhole Cost')
st.pyplot(firstfig)

VPI = mymodule.Vperfect(Pr_prior_POS_demo[0], value_array)
# VPI_list = list(map(lambda v: mymodule.f_Vperfect(Pr_prior_POS_demo, value_array, v), value_drill_DRYHOLE))
st.subheader(r'''$VOI_{perfect}$ ='''+str(VPI)+' - '+(str(vprior_INPUT_demo)+' = '+str(VPI-vprior_INPUT_demo)))
### END OF PASTE

with st.sidebar:
            
    # LOCATION OF THIS FILE (Carbonate Aquifer only to start?)
    uploaded_file = st.file_uploader("Choose a POS file",type=['csv'])

    if uploaded_file is not None:
        neg_upload_file = 'NEG'+str(uploaded_file.name[3:])
        file_path = '/Users/wtrainor/Library/CloudStorage/OneDrive-NREL/INGENIOUS/INGENIOUS_streamlit/assets/'
        st.write('Accompanying NEG file: ', file_path+neg_upload_file)
    
        df = pd.read_csv(uploaded_file)
        dfN = pd.read_csv(file_path+neg_upload_file)
        st.subheader('ML Nevada Data')
        st.write('File preview...')
        st.write(df.head())

        st.subheader('Choose attribute for VOI calculation')
        attribute0 = st.selectbox('What attributes would you like to calculate', df.columns) 
    # attribute0 = st.multiselect('What attributes would you like to calculate', df.columns,max_selections=2)
    
    # with st.echo(): # this prints out 
    # st.write("Update dry hole loss in sidebar.")
    # number = st.number_input(r'''$v_{a=drill}(\theta=Negative)$''')
    # st.write('The current number is ', number)

    #with st.spinner("Loading..."):
    #    time.sleep(5)
    #st.success("Done!")

st.title('Domain Likelihoods: thresholding distances to labels')

# uploaded_fileNEG = st.file_uploader("Choose a NEG file",type=['csv'])

if uploaded_file is not None:
    # df = pd.read_csv(uploaded_file)
    # dfN = pd.read_csv(file_path+neg_upload_file)
    # st.subheader('ML Nevada Data')
    # st.write('File preview...')
    # st.write(df.head())

    # st.write(df)
      
    if attribute0 is not None:
        st.write('You picked this attribute ', attribute0)

        x_cur = attribute0
    
        screen_att0 ='PosSite_Di'
        screen_att1 ='NegSite_Di'
        y_cur0 = 'GeodeticSt'  # hard code for now will come from multiselect
        # xmax_cur

        df_screen = df[df[x_cur]>-9999]
        df_screenN = dfN[dfN[x_cur]>-9999]
        st.write('dataframe is shape: {thesize}'.format(thesize=df_screen.shape))
        st.write('attribute stats ', df_screen[attribute0].describe())

        distance_meters = st.slider('Change likelihood by *screening* distance to positive label [km or meters??]', 
            10, int(np.max(df_screen['PosSite_Di'])-10), 800, step=100) # min, max, default
        # NEG_distance_meters = st.slider('Change likelihood by *screening* distance to negative label [km or meters??]', 
        #     10, int(np.max(df_screenN['NegSite_Di'])-10), int(np.median(df_screenN['NegSite_Di'])), step=1000)

        # round to make sure it rounds to nearest 10
        dfpair0 = df_screen[(df_screen['PosSite_Di'] <=round(distance_meters,-1))] 
        dfpair = dfpair0[dfpair0[y_cur0]>-9999] 
        # # # OJO : may want to keep this off until have it for NEG 
        dfpairN = df_screenN#[(df_screenN['NegSite_Di'] <=round(NEG_distance_meters,-1))] 
        st.subheader('Calculate & Display Likelihoods')
        mymodule.my_kdeplot(dfpair,x_cur,y_cur0,dfpairN)
        #waiting_condition = 1
        #while (waiting_condition):
        #    st.image('https://media.giphy.com/media/gu9XBXiz60HlO5p9Nz/giphy.gif')

        # waiting_condition = mymodule.my_kdeplot(dfpair,x_cur,y_cur0,y_cur1,waiting_condition)
        
        # split up if we want to test bandwidth 
        X_train, X_test, y_train, y_test = mymodule.make_train_test(dfpair,x_cur,y_cur0,dfpairN)
 
        # Likelihood via KDE estimate
        predictedLikelihood_pos, predictedLikelihood_neg, x_sampled, count_ij= mymodule.likelihood_KDE(X_train,X_test, y_train, y_test,x_cur,y_cur0)

        ## $Pr( \tilde{X} = \tilde{x}_j | X = x_i  )$
        st.write('*Given that we know the TRUE GEOTHERMAL OUTCOME (remember "$|$" stands for "given"), what is the likelihood of the interpretation ')
        st.latex(r'''
       {\tilde{x}_{j=0}}
        ''')
        st.write('We can compute this "empirical" likelihood with the counts of interpretations.')
        st.latex(r'''
        {Pr(\tilde{X}} =\tilde{x}_j | \Theta = \theta_i ) \approx  \frac{count_{ij}}{row\ sum = 15}
        ''')
        
        #Basic question: How far apart (different) are two distributions P and Q? Measured through distance & divergences
        #https://nobel.web.unc.edu/wp-content/uploads/sites/13591/2020/11/Distance-Divergence.pdf


        st.subheader('Change the Prior of POSITIVE Geothermal site')
        Pr_prior_POS = mymodule.Prior_probability_binary('Prior used in Posterior')
        
        # POSTERIOR from WGC
        # posterior = mymodule.Posterior_WGC()
        # POSTERIOR via_Naive_Bayes: Draw back here is ??? can I get the marginal out? 
        post_input, post_uniform = mymodule.Posterior_via_NaiveBayes(Pr_prior_POS,X_train, X_test, y_train, y_test, x_sampled, x_cur)

        value_array = mymodule.make_value_array(count_ij, profit_drill_pos= 1e6, cost_drill_neg = -1e6)
        st.write('value_array', value_array)

        #f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
        value_drill_DRYHOLE = np.linspace(100, -1e6,10)

        # This function can be called with multiple values of "dry hole"
        vprior_unif_out = mymodule.f_VPRIOR([1-Pr_prior_POS,Pr_prior_POS], value_array) #, value_drill_DRYHOLE[-1]       
        st.subheader('Should you enter the geothermal lottery?')
        st.subheader('Vprior '+str(vprior_unif_out))

        VPI = mymodule.Vperfect(Pr_prior_POS, value_array)
        st.subheader(r'''$VOI_{perfect}$ ='''+str(VPI))

        # Need a marginal estimate 
        # Calculate marg_input, marg_unif       
        # Passing unscale likelihood?
        Pr_Marg = mymodule.marginal(Pr_prior_POS, predictedLikelihood_pos, predictedLikelihood_neg)


        # VII_unif = mymodule.f_VIMPERFECT(post_uniform, value_array,Pr_UnifMarg)
        VII_input= mymodule.f_VIMPERFECT(post_input, value_array, Pr_Marg)
        st.subheader(r'''$V_{imperfect}$='''+str(VII_input))