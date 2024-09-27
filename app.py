import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
from PIL import Image
import requests
from io import BytesIO
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import babel.numbers
# import decimal
#import locale
#locale.setlocale( locale.LC_ALL, '' )

import mymodule

# 1 made empty repository on github
# 2 PyCharm Project from github: .py script that is github, made script & requirements.txt, commit & pushed
# 3 Log into streamlit, and app is there...

# PRIORS - > USER INPUT
st.header('Interactive Demonstration of Relationship between Value of Information and Prior Value')


#Code below plots the Decision Tree image from kmenon's github
# url = 'https://raw.githubusercontent.com/kmenon211/Geophysics-segyio-python/master/dtree.png'
# response = requests.get(url)
# image= Image.open(BytesIO(response.content))
# st.image(image, caption='Sample BinaryDecision Tree with Binary Geothermal Resource')


vprior_depth = np.array([1000,2000,3000,4000,5000,6000])

#st.write('What\'s the Prior Probability of a POSITIVE geothermal site?  $Pr(x=Positive)$')
#Pr_prior_POS_demo = mymodule.Prior_probability_binary() 


#### start of paste  -> CHANGE to input
count_ij = np.zeros((2,6))
value_array, value_array_df = mymodule.make_value_array(count_ij, profit_drill_pos= 15e6, cost_drill_neg = -1e6)
# # st.write('value_array', value_array)

value_drill_DRYHOLE = np.array([-1.9e6, -2.8e6, -4.11e6, -5.81e6, -7.9e6, -10.4e6])

vprior_depth = np.array([1000,2000,3000,4000,5000,6000])
value_drill_pos = value_drill_DRYHOLE
firstfig, ax = plt.subplots()
#firstfig1, axe = plt.subplots(1,2)
plt.plot(vprior_depth,value_drill_pos,'g.-', linewidth=5,label='$V_{prior}$')#, color = 'red')
plt.ylabel(r'Average Drilling Cost [\$]',fontsize=14)
plt.xlabel('Depth (m)', color='darkred',fontsize=14)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0f
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter('{x:0,.0f}')


# Code for table with decision outcomes defined by the user.
newValuedf1 = pd.DataFrame({
               "action": ['do nothing','drill'],
                
                "Hydrothermal Resource (positive)": [0,value_array_df.iloc[1,1]*10]}   
        )

# list = 
# idx= pd.Index(list)
# newValuedf.set_index(idx)
newValuedf1.style.set_properties(**{'font-size': '35pt'}) # this doesn't seem to work
 #bigdf.style.background_gradient(cmap, axis=1)\

# Code to input these values
original_title = '<p style="font-family:Courier; color:Black; font-size: 30px;"> Enter economic values for your decision</p>'
st.markdown(original_title, unsafe_allow_html=True)
edited_df = st.data_editor(newValuedf1,hide_index=True,use_container_width=True)

pos = np.float(edited_df[['Hydrothermal Resource (positive)']].values[1])
#neg = float(edited_df[['No Hydrothermal Resource (negative)']].values[1])
value_array, value_array_df = mymodule.make_value_array(count_ij, profit_drill_pos= pos, cost_drill_neg = -1e-6)


## Calculate Vprior
#f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
#value_drill_DRYHOLE = np.linspace(100, -1e6,10)
#Assigning values that match GEOPHIRES drilling costs.

#value_drill_DRYHOLE = np.array([10.4e6, 7.9e6, 5.81e6, 4.11e6, 2.8e6, 1.9e6])
#value_drill_DRYHOLE = np.array([-1.9e6, -2.8e6, -4.11e6, -5.81e6, -7.9e6, -10.4e6])

Pr_prior_POS_demo = mymodule.Prior_probability_binary() 
## Find Min Max for the Vprior Demo plot
vprior_INPUT_min = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[-1])  
vprior_INPUT_max = mymodule.f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[0])   
VPI_max = mymodule.Vperfect(Pr_prior_POS_demo, value_array,  value_drill_DRYHOLE[0])  

vprior_INPUT_demo_list = list(map(lambda vv: mymodule.f_VPRIOR([1-Pr_prior_POS_demo,Pr_prior_POS_demo], 
                                                              value_array,vv),value_drill_DRYHOLE))
st.subheader('$Pr(Success) = Pr(Geothermal=Positive)=$'+str(Pr_prior_POS_demo))  #Pr_prior_POS_demo[0]
st.subheader('$V_{prior} =$  best action given each weighted average')
# st.markdown("""
# <style>
# .big-latex {
#     font-size:60px !important;
# }
# </style>
# """, unsafe_allow_html=True)

#st.write('Average outcome, using $Pr(Success)$ ~ Prior probability')
# st.write(r'''$V_{prior} =  \max\limits_a \Sigma_{i=1}^2 Pr(\Theta = \theta_i)  v_a(\theta_i) \ \  \forall a $''')
#stuff = '''$V_{prior}=$'''
#st.markdown(stuff, unsafe_allow_html=True) #'<p class="big-latex"> stuff </p>'
st.write(r'''$V_{prior} = \max\limits_a 
            \begin{cases} Pr(positive) v_{drill}(positive) + Pr(negative)  v_{drill}(negative) &\text{if drill}\\
            Pr(positive) v_{nothing}(positive) + Pr(negative)  v_{nothing}(negative)  =0         &\text{if do nothing}\end{cases} $''')
#
# st.latex(r'''x = \begin{cases}
#    a &\text{if } b \\
#    c &\text{if } d \\
# \end{cases}''')
         
# st.write(r'''$\Theta$ =  Uncertain geologic parameter, $\theta_1$ = positive geothermal state, $\theta_2$ = negative geothermal state''')
# st.write(r'''$a =$  Action being taken (e.g. drill or do nothing)''')
#

# axins3 = inset_axes(ax, width="30%", height="30%", loc=2)
#st.write(np.mean(vprior_INPUT_demo_list), np.min(value_drill_DRYHOLE),(VPI_max+20))

# Plotting VOI
showVperfect = st.checkbox('Show Vperfect')

# Plotting Depth vs Value of Information

#showVperfect2 = st.checkbox('Show Vperfect')
firstfig2, ax1 = plt.subplots() # Plotting the VOI figure


ax1.plot(vprior_depth, vprior_INPUT_demo_list, 'g.-', linewidth=5,label='$V_{prior}$')
plt.ylabel(r'Average Outcome Value [\$]',fontsize=14)
plt.xlabel('Well Depth (m)', color='darkred',fontsize=14)



# Plotting text on the VOI plot

txtonplot = r'$v_{a=Drill}(\Theta=Positive) =$'
ax1.text(np.min(vprior_depth), value_array[-1,-1]*0.7, txtonplot+'\${:0,.0f}'.format(value_array[-1,-1]), 
        size=12, color='green',
         #va="baseline", ha="left", multialignment="left",
          horizontalalignment='left',
         verticalalignment='top')#, bbox=dict(fc="none"))

# Plotting the inset axes with drilling cost curve

if showVperfect:  
    VPIlist = list(map(lambda uu: mymodule.Vperfect(Pr_prior_POS_demo, value_array,uu),value_drill_DRYHOLE))
    # st.write('VPI',np.array(VPIlist),vprior_INPUT_demo_list)
    VOIperfect = np.maximum((np.array(VPIlist)-np.array(vprior_INPUT_demo_list)),np.zeros(len(vprior_INPUT_demo_list)))
    # VPI_list = list(map(lambda v: mymodule.f_Vperfect(Pr_prior_POS_demo, value_array, v), value_drill_DRYHOLE))
    ax1.plot(vprior_depth,VPIlist,'b', linewidth=5, alpha=0.5, label='$V_{perfect}$')
    ax1.plot(vprior_depth,VOIperfect,'b--', label='$VOI_{perfect}$')

plt.legend(loc=1)
plt.ylim([vprior_INPUT_min,value_array[-1,-1]*0.8]) # YLIM was (VPI_max+20)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0f
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_major_formatter('{x:0,.0f}')

axins1 = inset_axes(
    ax1,
    width="28%",  # width: 50% of parent_bbox width
    height="28%",  # height: 5%
    loc="center right",
)

axins1.plot(vprior_depth,value_drill_pos,'g.-', linewidth=5)#,color = 'red')

#plt.ylabel(r'Average Drilling Cost [\$]',fontsize=7)
plt.xlabel('Depth (m)', color='darkred',fontsize=7)
plt.title(r'Drilling Costs [\$]', fontsize = 7)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(True)
axins1.yaxis.set_major_formatter(formatter)
axins1.yaxis.set_major_formatter('${x:0,.0e}') #:0,.0f
axins1.xaxis.set_major_formatter(formatter)
axins1.xaxis.set_major_formatter('{x:0,.0f}')



#Code below plots the VOI plot
st.pyplot(firstfig2)

if showVperfect:  
    
    st.write('When you "know" when either subsurface condition occurs, you can pick the best ($\max\limits_a$) drilling altervative first ($v_a$).')
    st.write(r'''$V_{perfect} =  \Sigma_{i=1}^2 Pr(\Theta = \theta_i) \max\limits_a v_a(\theta_i) \ \  \forall a $''')
    st.write(r'''$VOI_{perfect} (Value \ of \ Information) = V_{perfect}-V_{prior}=$'''+str(VPIlist[0])+' - '+str(vprior_INPUT_demo_list[0]))

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) # Code to draw line to separate demo problem from main part
with st.sidebar:
    attribute0 = None        
    # LOCATION OF THIS FILE 
    uploaded_files = st.file_uploader("Upload two data files,namely a Positive Label file (\'POS_\' :fire:) & a Negative Label (\'NEG_\':thumbsdown:) file",type=['csv'],accept_multiple_files=True)
    
    count_neg= 0
    count_pos = 0
    if uploaded_files is not None and len(uploaded_files)==2:
        st.header('VOI APP')
        st.subheader('App Data')
        st.subheader('Choose attribute for VOI calculation')
        
        for uploaded_file in uploaded_files:
            # bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            if (uploaded_file.name[0:3]=='POS') :
                if ( count_pos == 1):
                    st.write('You didn\'t select a NEG file, try again')
                else:
                    pos_upload_file = uploaded_file
                    df = pd.read_csv(pos_upload_file)
            #       st.write('attribute0 is None',attribute0==None, not attribute0)
            #       if not attribute0:
                    attribute0 = st.selectbox('Which attribute would you like to explore?', df.columns) 
                    count_pos = count_pos + 1
            
            elif (uploaded_file.name[0:3]=='NEG'):
                if ( count_neg == 1):
                    st.write('You didn\'t select a POS file, try again')
                else:
                    neg_upload_file = uploaded_file
                    dfN = pd.read_csv(neg_upload_file)
                    count_neg = count_neg + 1       
            else:
                if ( uploaded_file.name[0:3] == 'NEG'):
                    st.write('You didn\'t select a POS file, try again')
                else:
                    st.write('You didn\'t select a NEG file, try again')

                
        

        if pos_upload_file.name[3:7] != neg_upload_file.name[3:7]:
                st.write('You aren\'t comparing data from the same region. STOP!')
        else:        
            st.write('POS File summary...')
            st.write(df.describe())        
            st.write('NEG File preview...')
            st.write(dfN.describe())   
        
    #with st.spinner("Loading..."):
    #    time.sleep(5)
    #st.success("Done!")
    else:
        st.write('please upload file')


# uploaded_fileNEG = st.file_uploader("Choose a NEG file",type=['csv'])
#st.write('uploaded_files==None attribute0==None', uploaded_files==None)
if uploaded_files is not None:
    st.title('Main App: ')
             
    if attribute0 is not None:
        st.title('You picked this attribute: '+attribute0)
        st.write('Thresholding distances to labels')

        x_cur = attribute0
    
        # screen_att0 ='PosSite_Di'
        # screen_att1 ='NegSite_Di'
        # if any(df.columns.str.contains('GeodeticStrainRate')):
        #     y_cur0 = 'GeodeticStrainRate'  # hard code for now will come from multiselect
        # elif any(df.columns.str.contains('geod_2ndinv_conus117_250m')):
        #     y_cur0 = 'geod_2ndinv_conus117_250m'
        # else:
        #     st.print('no know strain in df')

        df_screen = df[df[x_cur]>-9999]
        df_screenN = dfN[dfN[x_cur]>-9999]
        #st.write('dataframe is shape: {thesize}'.format(thesize=df_screenN.shape))
        #st.write('attribute stats ', df_screen[attribute0].describe())

        neg_site_col_name = 'NegSite_Distance' # I change the name when making csv 'tsite_dist_nvml_neg_conus117_250m' #
        distance_meters = st.slider('The '+neg_site_col_name+' Change likelihood by *screening* distance to positive label [meters]',
                                    10, int(np.max(df_screen['PosSite_Distance'])-10), int(np.max(df_screen['PosSite_Distance'].quantile(0.1))), step=100) # min, max, default
        # NEG_distance_meters = st.slider('Change likelihood by *screening* distance to negative label [km or meters??]', 
        #     10, int(np.max(df_screenN['NegSite_Di'])-10), int(np.median(df_screenN['NegSite_Di'])), step=1000)

        # round to make sure it rounds to nearest 10
        dfpair0 = df_screen[(df_screen['PosSite_Distance'] <=round(distance_meters,-1))] 
        
        dfpair = dfpair0[dfpair0[x_cur]>-9999] 
        # # # OJO : may want to keep this off until have it for NEG 
        dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <=round(distance_meters,-1))] 
        
        if np.shape(dfpairN)[0]==0:
            st.write('using Q1 distance for Negative sites')
            dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <= np.percentile(df_screenN[neg_site_col_name ],10))] 
        
        st.subheader('Calculate & Display Likelihoods')
        st.write('We can compute this "empirical" likelihood with the counts of interpretations.')
       
        #waiting_condition = 1
        #while (waiting_condition):
        #    st.image('https://media.giphy.com/media/gu9XBXiz60HlO5p9Nz/giphy.gif')

        # waiting_condition = mymodule.my_kdeplot(dfpair,x_cur,y_cur0,y_cur1,waiting_condition)
        
        # split up if we want to test bandwidth 
        X_train, X_test, y_train, y_test = mymodule.make_train_test(dfpair,x_cur,dfpairN)
 
        best_params, accuracy = mymodule.optimal_bin(X_train, y_train)

        # Likelihood via KDE estimate
        predictedLikelihood_pos, predictedLikelihood_neg, x_sampled, count_ij= mymodule.likelihood_KDE(X_train,X_test, y_train, y_test,x_cur, best_params)
      
        st.write(':blue['+r'''$Pr(\Theta = \theta_i)$'''+'] in posterior')
        Pr_prior_POS = mymodule.Prior_probability_binary('Prior used in Posterior')

        st.subheader('~:blue[Prior]:point_up_2: x Likelhood :arrow_heading_up:')            
        # New plot for normalized likelihood: Modeled after Likelihood via KDE estimate
        mymodule.Scaledlikelihood_KDE(Pr_prior_POS,predictedLikelihood_neg, predictedLikelihood_pos,X_train,X_test, y_train, y_test,x_cur,x_sampled, best_params)
        
        st.header('How much is this imperfect data worth?')
        st.subheader(':point_down: :violet[Posterior]~:blue[Prior]:point_up_2: x Likelhood :arrow_heading_up:')
        # POSTERIOR via_Naive_Bayes: Draw back here the marginal not using scaled likelihood..
        post_input, post_uniform = mymodule.Posterior_via_NaiveBayes(Pr_prior_POS,X_train, X_test, y_train, y_test, x_sampled, x_cur)
             
        # # DO NOT USEmymodule.marginal( because it's passing unscaled likelihood!!!)
        # # Pr_Marg = mymodule.marginal(Pr_prior_POS, predictedLikelihood_pos, predictedLikelihood_neg, x_sampled)
        Pr_InputMarg, Pr_UnifMarg, Prm_d_Input, Prm_d_Uniform = mymodule.Posterior_by_hand(Pr_prior_POS,predictedLikelihood_pos, predictedLikelihood_neg, x_sampled)
        mymodule.Posterior_Marginal_plot(Prm_d_Input, Prm_d_Uniform, Pr_InputMarg, x_cur, x_sampled) # WAS inputting: post_input, post_uniform, Pr_Marg, x_cur, x_sampled)

        # # # # # # VALUE OUTCOMES # # # # # # # # # #
        newValuedf = pd.DataFrame({
               "action": ['do nothing','drill'],
                "No Hydrothermal Resource (negative)": [0, value_array_df.iloc[1,0]*10],
                "Hydrothermal Resource (positive)": [0,value_array_df.iloc[1,1]*10]}   
        )

        # list = 
        # idx= pd.Index(list)
        # newValuedf.set_index(idx)
        newValuedf.style.set_properties(**{'font-size': '35pt'}) # this doesn't seem to work
        #bigdf.style.background_gradient(cmap, axis=1)\

        # Code to be written to input these values
        original_title = '<p style="font-family:Courier; color:Green; font-size: 30px;"> Enter economic values for your decision</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        edited_df = st.data_editor(newValuedf,hide_index=True,use_container_width=True)

        pos = np.float(edited_df[['Hydrothermal Resource (positive)']].values[1])
        neg = np.float(edited_df[['No Hydrothermal Resource (negative)']].values[1])

        value_array, value_array_df = mymodule.make_value_array(count_ij, profit_drill_pos= pos, cost_drill_neg = neg) # Karthik Changed here to reflect new values
        #st.write('value_array', value_array)

        #f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
        value_drill_DRYHOLE = np.linspace(100, -1e6,10)

        # This function can be called with multiple values of "dry hole"
        vprior_unif_out = mymodule.f_VPRIOR([1-Pr_prior_POS,Pr_prior_POS], value_array) #, value_drill_DRYHOLE[-1]       
                       
        #st.subheader(r'''$V_{prior}$ '''+'${:0,.0f}'.format(vprior_unif_out).replace('$-','-$'))

        VPI = mymodule.Vperfect(Pr_prior_POS, value_array)
        # st.subheader(r'''$VOI_{perfect}$ ='''+str(locale.currency(VPI, grouping=True )))
        #st.subheader('Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI).replace('$-','-$'))

        # VII_unif = mymodule.f_VIMPERFECT(post_uniform, value_array,Pr_UnifMarg)
        VII_input = mymodule.f_VIMPERFECT(Prm_d_Input, value_array, Pr_InputMarg)
        VII_unifPrior = mymodule.f_VIMPERFECT(Prm_d_Uniform, value_array, Pr_UnifMarg)
        
        st.latex(r'''\color{purple} Pr( \Theta = \theta_i | X =x_j ) = \color{blue}
        \frac{Pr(\Theta = \theta_i ) \color{black} Pr( X=x_j | \Theta = \theta_i )}{\color{orange} Pr (X=x_j)}''')
        
        # st.write('Using these $v_a(\Theta)$',value_array_df)
        
        # list = 
        # idx= pd.Index(list)
        # newValuedf.set_index(idx)
        #newValuedf.style.set_properties(**{'font-size': '35pt'}) # this doesn't seem to work
        #bigdf.style.background_gradient(cmap, axis=1)\

        # Code to be written to input these values
        #original_title = '<p style="font-family:Courier; color:Green; font-size: 30px;"> Enter economic values for your decision</p>'
        #st.markdown(original_title, unsafe_allow_html=True)
        #edited_df = st.data_editor(newValuedf,hide_index=True,use_container_width=True)
        
        #st.data_editor(value_array_df,
                         #column_config={
                        #"positive": st.column_config.NumberColumn(
                         #"Price (in USD)",
                         #help="Change the profit in USD",
                         #min_value=-1e12,
                         #max_value=1e12,
                         #step=1e3,
                         #format="$%d",
                         #)
                     #},
             #hide_index=True,
         #)
        
        st.subheader(r'''$V_{imperfect}$='''+'${:0,.0f}'.format(VII_input).replace('$-','-$'))
        st.subheader('Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI).replace('$-','-$'))
        # st.write('with uniform marginal', locale.currency(VII_unifMarginal, grouping=True ))
        # st.write('with uniform Prior', '${:0,.0f}'.format(VII_unifPrior).replace('$-','-$'))
        
        MI_post, NMI_post = mymodule.f_MI(Prm_d_Input,Pr_InputMarg)
        #Basic question: How far apart (different) are two distributions P and Q? Measured through distance & divergences
        #https://nobel.web.unc.edu/wp-content/uploads/sites/13591/2020/11/Distance-Divergence.pdf
        # st.write('Mutual Information:', MI_post)
        # st.write('Normalized Mutual Information:', NMI_post)
        # st.write(accuracy,(VII_input,MI_post,accuracy)) #['bandwidth']
        dataframe4clipboard = pd.DataFrame([[VII_input,NMI_post,accuracy]])#,  columns=['VII','NMI','accuracy'])
        #st.write(dataframe4clipboard)
       #dataframe4clipboard.to_clipboard(excel=True,index=False)

    else: 
        st.write("Please upload data files on left")
else:
    st.write("Please upload any data.")
        