import random
import time;
import datetime
import tkinter.messagebox


from tkinter import*

root= Tk()
root.iconbitmap(r"E:\ALL COURSES\ipopoola PYTHON PROJECTS\ipopoola restaurant management system\Graphicloads-Colorful-Long-Shadow-Restaurant.ico")
root.geometry("1300x700+0+0")
root.title("ipopoola Restaurant Systems")
root.configure(background = "Yellow")

Tops = Frame(root, bg='Yellow', bd=20, pady=5, relief= GROOVE)
Tops.pack(side=TOP)


lblTitle = Label(Tops, font= ('Monteserrat',40), 
	text = "ipopoola Restaurant Management System", 
	bd=21, bg="yellow", fg='Black', justify='center')

lblTitle.grid(row=0, column=0)




ReceiptCal_F = Frame(root, bg='yellow', bd=8,  relief= GROOVE)
ReceiptCal_F.pack(side=RIGHT)

Buttons_F=Frame(
ReceiptCal_F, bg='yellow', bd=3,  relief= GROOVE)
Buttons_F.pack(side=BOTTOM)


Cal_F=Frame(
ReceiptCal_F, bg='yellow', bd=6,  relief= GROOVE)
Cal_F.pack(side=TOP)



Receipt_F=Frame(
ReceiptCal_F, bg='yellow', bd=3,  relief= GROOVE)
Receipt_F.pack(side=BOTTOM)


MenuFrame = Frame(root, bg='yellow', bd=10, relief= RIDGE)
MenuFrame.pack(side=LEFT)


Cost_F=Frame(
MenuFrame, bg='yellow', bd=4)
Cost_F.pack(side=BOTTOM)




Drinks_F=Frame(
MenuFrame, bg='yellow', bd=10)
Drinks_F.pack(side=TOP)


Drinks_F=Frame(
MenuFrame, bg='yellow', bd=10, relief= RIDGE)
Drinks_F.pack(side=LEFT)



Cake_F=Frame(
MenuFrame, bg='yellow', bd=10, relief= RIDGE )
Cake_F.pack(side=RIGHT)

#================================================Variables======================================

var1=IntVar()
var2=IntVar()
var3=IntVar()
var4=IntVar()
var5=IntVar()
var6=IntVar()
var7=IntVar()
var8=IntVar()
var9=IntVar()
var10=IntVar()
var11=IntVar()
var12=IntVar()
var13=IntVar()
var14=IntVar()
var15=IntVar()
var16=IntVar()


DateofOrder = StringVar()
Receipt_Ref = StringVar()
PaidTax = StringVar()
SubTotal= StringVar()
TotalCost = StringVar()
CostofCakes = StringVar()
CostofDrinks = StringVar()
ServiceCharge = StringVar()

text_Input= StringVar()
operator=""

E_Latta=StringVar()
E_Espresso=StringVar()
E_Iced_Latta=StringVar()
E_Vale_Coffe=StringVar()
E_Cappuccino=StringVar()
E_African_Coffee=StringVar()
E_American_Coffee=StringVar()
E_Iced_Cappuccino=StringVar()



E_School_Cake=StringVar()
E_Sunny_AO_Cake=StringVar()
E_Jonathan_YO_Cake=StringVar()
E_West_African_Cake=StringVar()
E_Lagos_Chocolate_Cake=StringVar()
E_Kilburn_Chocolate_Cake=StringVar()
E_Carlton_Hill_Chocolate_Cake=StringVar()
E_Queen_Park_Chocolate_Cake=StringVar()


E_Latta.set("0")
E_Espresso.set("0")
E_Iced_Latta.set("0")
E_Vale_Coffe.set("0")
E_Cappuccino.set("0")
E_African_Coffee.set("0")
E_American_Coffee.set("0")
E_Iced_Cappuccino.set("0")



E_School_Cake.set("0")
E_Sunny_AO_Cake.set("0")
E_Jonathan_YO_Cake.set("0")
E_West_African_Cake.set("0")
E_Lagos_Chocolate_Cake.set("0")
E_Kilburn_Chocolate_Cake.set("0")
E_Carlton_Hill_Chocolate_Cake.set("0")
E_Queen_Park_Chocolate_Cake.set("0")




DateofOrder.set(time.strftime("%d/%m/%Y"))


####################################EXIT FUNCTION DECLARED########################################################
def iExit():
	iExit = tkinter.messagebox.askyesno("Exit Restaurant System", "CONFIRM IF YOU WANT TO EXIT")
	if iExit > 0:
		root.destroy()
		return

def Reset():

	PaidTax.set("0")
	SubTotal.set("0")
	TotalCost.set("0")
	CostofCakes.set("0")
	CostofDrinks.set("0")
	ServiceCharge.set("0")
	txtReceipt.delete("1.0", END)



	E_Latta.set("0")
	E_Espresso.set("0")
	E_Iced_Latta.set("0")
	E_Vale_Coffe.set("0")
	E_Cappuccino.set("0")
	E_African_Coffee.set("0")
	E_American_Coffee.set("0")
	E_Iced_Cappuccino.set("0")



	E_School_Cake.set("0")
	E_Sunny_AO_Cake.set("0")
	E_Jonathan_YO_Cake.set("0")
	E_West_African_Cake.set("0")
	E_Lagos_Chocolate_Cake.set("0")
	E_Kilburn_Chocolate_Cake.set("0")
	E_Carlton_Hill_Chocolate_Cake.set("0")
	E_Queen_Park_Chocolate_Cake.set("0")





	var1.set("0")
	var2.set("0")
	var3.set("0")
	var4.set("0")
	var5.set("0")
	var6.set("0")
	var7.set("0")
	var8.set("0")
	var9.set("0")
	var10.set("0")
	var11.set("0")
	var12.set("0")
	var13.set("0")
	var14.set("0")
	var15.set("0")
	var16.set("0")


	txtLatta.configure(state = DISABLED)
	txtEspresso.configure(state = DISABLED)
	txtIced_Latte.configure(state = DISABLED)
	txtVale_Coffee.configure(state = DISABLED)
	txtCappuccino.configure(state = DISABLED)
	txtAfrican_Coffee.configure(state = DISABLED)
	txtAmerican_Coffee.configure(state = DISABLED)
	txtIced_Cappuccino.configure(state = DISABLED)
	txtSchool_Cake.configure(state = DISABLED)
	txtSunny_AO_Cake.configure(state = DISABLED)
	txtJonathan_YO_Cake.configure(state = DISABLED)
	txtWest_African_Cake.configure(state = DISABLED)
	txtLagos_Chocolate_Cake.configure(state = DISABLED)
	txtKilburn_Chocolate_Cake.configure(state = DISABLED)
	txtCarlton_Hill_Chocolate_Cake.configure(state = DISABLED)
	txtQueen_Park_Chocolate_Cake.configure(state = DISABLED)


def CostofItem():
	Item1=float(E_Latta.get())
	Item2=float(E_Espresso.get())
	Item3=float(E_Iced_Latta.get())
	Item4=float(E_Vale_Coffe.get())
	Item5=float(E_Cappuccino.get())
	Item6=float(E_African_Coffee.get())
	Item7=float(E_American_Coffee.get())
	Item8=float(E_Iced_Cappuccino.get())

	Item9=float(E_School_Cake.get())
	Item10=float(E_Sunny_AO_Cake.get())
	Item11=float(E_Jonathan_YO_Cake.get())
	Item12=float(E_West_African_Cake.get())
	Item13=float(E_Lagos_Chocolate_Cake.get())
	Item14=float(E_Kilburn_Chocolate_Cake.get())
	Item15=float(E_Carlton_Hill_Chocolate_Cake.get())
	Item16=float(E_Queen_Park_Chocolate_Cake.get())


	PriceofDrinks = (Item1 * 1.2) + (Item2 * 1.99) + (Item3 * 2.05) \
		+ (Item4 * 1.89) + (Item5 * 1.99) + (Item6 * 2.99) + (Item7 * 2.39) + (Item8 * 1.29)

	PriceofCakes = (Item9 * 1.35) + (Item10 * 2.2) + (Item11 * 1.99) \
		+ (Item12 * 1.49) + (Item13 * 1.8) + (Item14 * 1.67) + (Item15 * 1.6) + (Item16 * 1.99)


	DrinksPrice = "$", str('%.2f'%(PriceofDrinks))
	CakesPrice = "$", str('%.2f'%(PriceofCakes))
	CostofCakes.set(CakesPrice)
	CostofDrinks.set(DrinksPrice)
	SC = "$", str('%.2f'%(1.59))
	ServiceCharge.set(SC)


	SubTotalofITEMS = "$", str('%.2f'%(PriceofDrinks + PriceofCakes + 1.59))
	SubTotal.set(SubTotalofITEMS)


	Tax = "$", str('%.2f'%((PriceofDrinks + PriceofCakes + 1.59)* 0.15))
	PaidTax.set(Tax)
	TT=((PriceofDrinks + PriceofCakes + 1.59)* 0.15)
	TC =  "$", str('%.2f'%(PriceofDrinks + PriceofCakes + 1.59 + TT)) 
	TotalCost.set(TC)

def chkLatta():
	if (var1.get() == 1):
		txtLatta.configure(state=NORMAL)
		txtLatta.focus()
		txtLatta.delete('0', END)
		E_Latta.set("")


	elif(var1.get() == 0):
		txtLatta.configure(state=DISABLED)
		E_Latta.set("0")



def chkEspresso():
	if (var2.get() == 1):
		txtEspresso.configure(state=NORMAL)
		txtEspresso.focus()	
		E_Espresso.set("")
	elif var2.get() == 0:
		txtEspresso.configure(state=DISABLED)
		E_Espresso.set("0")


def chkIced_Latte():
	if (var3.get() == 1):
		txtIced_Latte.configure(state=NORMAL)
		txtIced_Latte.delete('0',END)
		txtIced_Latte.focus()
		
	elif var3.get() == 0:
		txtIced_Latte.configure(state=DISABLED)
		E_Iced_Latta.set("0")


def chkVale_Coffee():
	if (var4.get() == 1):
		txtVale_Coffee.configure(state=NORMAL)
		txtVale_Coffee.delete('0',END)
		txtVale_Coffee.focus()
		
	elif var4.get() == 0:
		txtVale_Coffee.configure(state=DISABLED)
		E_Vale_Coffe.set("0")


def chkCappuccino():
	if (var5.get() == 1):
		txtCappuccino.configure(state=NORMAL)
		txtCappuccino.delete('0',END)
		txtCappuccino.focus()
		
	elif var5.get() == 0:
		txtCappuccino.configure(state=DISABLED)
		E_Cappuccino.set("0")


def chkAfrican_Coffee():
	if (var6.get() == 1):
		txtAfrican_Coffee.configure(state=NORMAL)
		txtAfrican_Coffee.delete('0',END)
		txtAfrican_Coffee.focus()
		
	elif var6.get() == 0:
		txtAfrican_Coffee.configure(state=DISABLED)
		E_African_Coffee.set("0")


def chkAmerican_Coffee():
	if (var7.get() == 1):
		txtAmerican_Coffee.configure(state=NORMAL)
		txtAmerican_Coffee.delete('0',END)
		txtAmerican_Coffee.focus()
		
	elif var7.get() == 0:
		txtAmerican_Coffee.configure(state=DISABLED)
		E_American_Coffee.set("0")


def chkIced_Cappuccino():
	if (var8.get() == 1):
		txtIced_Cappuccino.configure(state=NORMAL)
		txtIced_Cappuccino.delete('0',END)
		txtIced_Cappuccino.focus()
		
	elif var8.get() == 0:
		txtIced_Cappuccino.configure(state=DISABLED)
		E_Iced_Cappuccino.set("0")


def chkSchool_Cake():
	if (var9.get() == 1):
		txtSchool_Cake.configure(state=NORMAL)
		txtSchool_Cake.delete('0',END)
		txtSchool_Cake.focus()
		
	elif var9.get() == 0:
		txtSchool_Cake.configure(state=DISABLED)
		E_School_Cake.set("0")


def chkSunny_AO_Cake():
	if (var10.get() == 1):
		txtSunny_AO_Cake.configure(state=NORMAL)
		txtSunny_AO_Cake.delete('0',END)
		txtSunny_AO_Cake.focus()
		
	elif var10.get() == 0:
		txtSunny_AO_Cake.configure(state=DISABLED)
		E_Sunny_AO_Cake.set("0")


def chkJonathan_YO_Cake():
	if (var11.get() == 1):
		txtJonathan_YO_Cake.configure(state=NORMAL)
		txtJonathan_YO_Cake.delete('0',END)
		txtJonathan_YO_Cake.focus()
		
	elif var11.get() == 0:
		txtJonathan_YO_Cake.configure(state=DISABLED)
		E_Jonathan_YO_Cake.set("0")



def chkWest_African_Cake():
	if (var12.get() == 1):
		txtWest_African_Cake.configure(state=NORMAL)
		txtWest_African_Cake.delete('0',END)
		txtWest_African_Cake.focus()
		
	elif var12.get() == 0:
		txtWest_African_Cake.configure(state=DISABLED)
		E_West_African_Cake.set("0")



def chkLagos_Chocolate_Cake():
	if (var13.get() == 1):
		txtLagos_Chocolate_Cake.configure(state=NORMAL)
		txtLagos_Chocolate_Cake.delete('0',END)
		txtLagos_Chocolate_Cake.focus()
		
	elif var13.get() == 0:
		txtLagos_Chocolate_Cake.configure(state=DISABLED)
		E_Lagos_Chocolate_Cake.set("0")


def chkKilburn_Chocolate_Cake():
	if (var14.get() == 1):
		txtKilburn_Chocolate_Cake.configure(state=NORMAL)
		txtKilburn_Chocolate_Cake.delete('0',END)
		txtKilburn_Chocolate_Cake.focus()
		
	elif var14.get() == 0:
		txtKilburn_Chocolate_Cake.configure(state=DISABLED)
		E_Kilburn_Chocolate_Cake.set("0")



def chkCarlton_Hill_Cake():
	if (var15.get() == 1):
		txtCarlton_Hill_Chocolate_Cake.configure(state=NORMAL)
		txtCarlton_Hill_Chocolate_Cake.delete('0',END)
		txtCarlton_Hill_Chocolate_Cake.focus()
		
	elif var15.get() == 0:
		txtCarlton_Hill_Chocolate_Cake.configure(state=DISABLED)
		E_Carlton_Hill_Chocolate_Cake.set("0")





def chkQueen_Park_Cake():
	if (var16.get() == 1):
		txtQueen_Park_Chocolate_Cake.configure(state=NORMAL)
		txtQueen_Park_Chocolate_Cake.delete('0',END)
		txtQueen_Park_Chocolate_Cake.focus()
		
	elif var16.get() == 0:
		txtQueen_Park_Chocolate_Cake.configure(state=DISABLED)
		E_Queen_Park_Chocolate_Cake.set("0")



def Receipt ():
	txtReceipt.delete("1.0", END)
	x = random.randint(10903, 609235)
	randomRef = str(x)
	Receipt_Ref.set("BILL" + randomRef)


	txtReceipt.insert(END, 'Receipt Ref: \t\t\t' + Receipt_Ref.get() + '\t' + DateofOrder.get()+ "\n")
	txtReceipt.insert(END, 'Item: \t\t\t' + "Cost of Items \n")
	txtReceipt.insert(END, 'Latta: \t\t\t' + E_Latta.get()+ "\n")
	txtReceipt.insert(END, 'Espresso: \t\t\t' + E_Espresso.get()+ "\n")
	txtReceipt.insert(END, 'Iced_Latta: \t\t\t' + E_Iced_Latta.get()+ "\n")
	txtReceipt.insert(END, 'Vale_Coffe: \t\t\t' + E_Vale_Coffe.get()+ "\n")
	txtReceipt.insert(END, 'Cappucino: \t\t\t' + E_Cappuccino.get()+ "\n")
	txtReceipt.insert(END, 'African_Coffee: \t\t\t' + E_African_Coffee.get()+ "\n")
	txtReceipt.insert(END, 'American_Coffee: \t\t\t' + E_American_Coffee.get()+ "\n")
	txtReceipt.insert(END, 'Iced_Cappucino: \t\t\t' + E_School_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Sunny_AO_Cake: \t\t\t' + E_Sunny_AO_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Jonathan_YO_Cake: \t\t\t' + E_Jonathan_YO_Cake.get()+ "\n")
	txtReceipt.insert(END, 'West_African_Cake: \t\t\t' + E_West_African_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Lagos_Chocolate_Cake: \t\t\t' + E_Lagos_Chocolate_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Kilburn_Chocolate_Cake: \t\t\t' + E_Kilburn_Chocolate_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Carlton Hill Chocolate Cake: \t\t\t' + E_Carlton_Hill_Chocolate_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Queens Park Chocolate Cake: \t\t\t' + E_Queen_Park_Chocolate_Cake.get()+ "\n")
	txtReceipt.insert(END, 'Cost of Drinks: \t\t\t' + CostofDrinks.get()+ '\nTax Paid:\t\t\t\t' + PaidTax.get()+ "\n")
	txtReceipt.insert(END, 'Cost of Cakes: \t\t\t' + CostofCakes.get()+ '\nSubTotal:\t\t\t\t' + str(SubTotal.get())+ "\n")
	txtReceipt.insert(END, 'Service Charge: \t\t\t' + ServiceCharge.get()+ '\nTotal Cost:\t\t\t\t' + str(TotalCost.get()))



#================================================DRINKS======================================
Latta=Checkbutton(Drinks_F, text="Latta", variable=var1, onvalue=1, offvalue=0,
	font=('Monteserrat', 16), bg='yellow', 
	command = chkLatta).grid(row=0, sticky=W)


Espresso=Checkbutton(Drinks_F, text="Espresso", variable=var2, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkEspresso).grid(row=1, sticky=W)


Iced_Latta=Checkbutton(Drinks_F, text="Iced Latta", variable=var3, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkIced_Latte).grid(row=2, sticky=W)

Vale_Coffee=Checkbutton(Drinks_F, text="Vale Coffee", variable=var4, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkVale_Coffee).grid(row=3, sticky=W)

Cappucino=Checkbutton(Drinks_F, text="Cappucino", variable=var5, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkCappuccino).grid(row=4, sticky=W)

African_Coffee=Checkbutton(Drinks_F, text="African Coffee", variable=var6, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkAfrican_Coffee).grid(row=5, sticky=W)

American_Coffee=Checkbutton(Drinks_F, text="American Coffee", variable=var7, onvalue=1, 
	offvalue=0,	font=('Monteserrat', 16), bg='yellow', 
	command = chkAmerican_Coffee).grid(row=6, sticky=W)

Iced_Cappucino=Checkbutton(Drinks_F, text="Iced Cappucino", variable=var8, onvalue=1,
	offvalue=0,	font=('Monteserrat',16), bg='yellow' ,
	command = chkIced_Cappuccino).grid(row=7, sticky=W)


#============================================ENTRY BOX FOR DRINKS==========================================
txtLatta = Entry(Drinks_F, font=('arial', 16), textvariable=E_Latta, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtLatta.grid(row=0, column=1)

txtEspresso = Entry(Drinks_F, font=('arial', 16), textvariable=E_Espresso, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtEspresso.grid(row=1, column=1)


txtIced_Latte = Entry(Drinks_F, font=('arial', 16), textvariable=E_Iced_Latta, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtIced_Latte.grid(row=2, column=1)

txtVale_Coffee = Entry(Drinks_F, font=('arial', 16), textvariable=E_Vale_Coffe, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtVale_Coffee.grid(row=3, column=1)

txtCappuccino = Entry(Drinks_F, font=('arial', 16), textvariable=E_Cappuccino, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtCappuccino.grid(row=4, column=1)


txtAfrican_Coffee = Entry(Drinks_F, font=('arial', 16), textvariable=E_African_Coffee, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtAfrican_Coffee.grid(row=5, column=1)

txtAmerican_Coffee = Entry(Drinks_F, font=('arial', 16), textvariable=E_American_Coffee, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtAmerican_Coffee.grid(row=6, column=1)


txtIced_Cappuccino = Entry(Drinks_F, font=('arial', 16), textvariable=E_Iced_Cappuccino, bd=8, width=6, 
	justify=LEFT, state= DISABLED)
txtIced_Cappuccino.grid(row=7, column=1)



#============================================Cakes==========================================

SchoolCake=Checkbutton(Cake_F, text="School Cake\t\t\t", variable=var9,
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkSchool_Cake).grid(row=0, sticky=W)


Sunny_AO_Cake=Checkbutton(Cake_F, text="Espresso", variable=var10, 
	onvalue=1, offvalue=0,	font=('arial',16, ), bg='yellow', 
	command = chkSunny_AO_Cake).grid(row=1, sticky=W)


Jonathan_YO_Cake=Checkbutton(Cake_F, text="Jonathan O Cake", variable=var11, 
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkJonathan_YO_Cake).grid(row=2, sticky=W)

West_African_Cake=Checkbutton(Cake_F, text="West African Cake", variable=var12, 
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkWest_African_Cake).grid(row=3, sticky=W)

Lagos_Chocolate_Cake=Checkbutton(Cake_F, text="Lagos Chocolate Cake", variable=var13, 
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkLagos_Chocolate_Cake).grid(row=4, sticky=W)

Kilburn_Chocolate_Cake=Checkbutton(Cake_F, text="Kilburn Chocolate Cake", variable=var14, 
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkKilburn_Chocolate_Cake).grid(row=5, sticky=W)

Carlton_Hill_Cake=Checkbutton(Cake_F, text="Carlton Hill Cake", variable=var15, 
	onvalue=1, offvalue=0, font=('arial', 16, ), bg='yellow', 
	command = chkCarlton_Hill_Cake).grid(row=6, sticky=W)

Queen_Park_Cake=Checkbutton(Cake_F, text="Queen's Park Chocolate", variable=var16, 
	onvalue=1, offvalue=0,	font=('arial', 16, ), bg='yellow', 
	command = chkQueen_Park_Cake).grid(row=7, sticky=W)


#============================================ENTRY BOX FOR CAKES==========================================
txtSchool_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED, textvariable=E_School_Cake)
txtSchool_Cake.grid(row=0, column=1)

txtSunny_AO_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED,  textvariable=E_Sunny_AO_Cake)
txtSunny_AO_Cake.grid(row=1, column=1)


txtJonathan_YO_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED,  textvariable=E_Jonathan_YO_Cake)
txtJonathan_YO_Cake.grid(row=2, column=1)

#txtVale_Coffee = Entry(Drinks_F, font=('arial', 16,'bold'), bd=3, width=6, justify=LEFT, state= DISABLED)
#txtVale_Coffee.grid(row=3, column=1)

txtWest_African_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED,  textvariable=E_West_African_Cake)
txtWest_African_Cake.grid(row=3, column=1)

txtLagos_Chocolate_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED, textvariable=E_Lagos_Chocolate_Cake)
txtLagos_Chocolate_Cake.grid(row=4, column=1)


txtKilburn_Chocolate_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED,  textvariable=E_Kilburn_Chocolate_Cake)
txtKilburn_Chocolate_Cake.grid(row=5, column=1)

txtCarlton_Hill_Chocolate_Cake = Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED, textvariable=E_Carlton_Hill_Chocolate_Cake)
txtCarlton_Hill_Chocolate_Cake.grid(row=6, column=1)


txtQueen_Park_Chocolate_Cake= Entry(Cake_F, font=('arial', 16), bd=8, width=6, justify=LEFT, 
	state= DISABLED,  textvariable=E_Queen_Park_Chocolate_Cake)
txtQueen_Park_Chocolate_Cake.grid(row=7, column=1)

#===========================================Total Cost=============================================
lblCostofDrinks = Label(Cost_F, font= ('Banchschrift', 14), text = "Cost of Drinks\t ", 
						 bg="yellow", fg='Black', )
lblCostofDrinks.grid(row=0, column=0, sticky=W)
txtCostofDrinks = Entry(Cost_F, bg= "white", insertwidth=2, bd=7,
						 font=('arial', 14 ), textvariable=CostofDrinks, justify= RIGHT )
txtCostofDrinks.grid(row=0, column=1)


lblCostofCakes = Label(Cost_F, font= ('Banchschrift', 14), text = "Cost of Cakes\t ", 
						 bg="yellow", fg='Black', )
lblCostofCakes.grid(row=1, column=0, sticky=W)
txtCostofCakes = Entry(Cost_F, bg= "white", insertwidth=2, bd=7,
						 font=('arial', 14 ) , textvariable=CostofCakes, justify= RIGHT )
txtCostofCakes.grid(row=1, column=1)


lblServiceCharge = Label(Cost_F, font= ('Banchschrift', 14), text = "Service Charge\t ", 
						bg="yellow", fg='Black')

lblServiceCharge.grid(row=2, column=0, sticky=W)
txtServiceCharge = Entry(Cost_F, bg= "white", bd=7, 
						font=('arial', 14 ),  textvariable=ServiceCharge, justify= RIGHT )
txtServiceCharge.grid(row=2, column=1)


#===========================================Payment Information=============================================
lblPaidTax = Label(Cost_F, font= ('Banchschrift', 14), text = "\tPaid Tax\t ", bd=7, 
						 bg="yellow", fg='Black', )
lblPaidTax.grid(row=0, column=2, sticky=W)
txtPaidTax = Entry(Cost_F, bg= "white", insertwidth=2, bd=7,
						 font=('arial', 14 ),  textvariable=PaidTax, justify= RIGHT )
txtPaidTax.grid(row=0, column=3)


lblSubTotal = Label(Cost_F, font= ('Banchschrift', 14), text = "\tSub Total ", bd=7, 
						 bg="yellow", fg='Black', )
lblSubTotal.grid(row=1, column=2, sticky=W)
txtSubTotal = Entry(Cost_F, bg= "white", insertwidth=2, bd=7,
						 font=('arial', 14 ),  textvariable=SubTotal, justify= RIGHT )
txtSubTotal.grid(row=1, column=3)


lblTotalCost = Label(Cost_F, font= ('Banchschrift', 14), text = "\tTotal Cost ", bd=7,
						bg="yellow", fg='Black')
lblTotalCost.grid(row=2, column=2, sticky=W)
txtTotalCost  = Entry(Cost_F, bg= "white", bd=7, 
						font=('arial', 14 ),  textvariable=TotalCost, insertwidth=2, justify= RIGHT )
txtTotalCost.grid(row=2, column=3)


#===========================================Receipt===================================================
txtReceipt = Text(Receipt_F, width=46, height=12, bg= "white", bd=4, font=('arial', 12, ))
txtReceipt.grid(row=0, column=0)




#===========================================Buttons===================================================
btnTotal = Button(Buttons_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), 
	width=4,  text="Total",	bg="yellow", command = CostofItem).grid(row=0, column=0)

btnReceipt = Button(Buttons_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), 
	width=4,  text="Receipt", bg="yellow", command=Receipt).grid(row=0, column=1)

btnReset = Button(Buttons_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="Reset",	bg="yellow", command=Reset).grid(row=0, column=2)

btnExit = Button(Buttons_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="Exit", bg="yellow", command=iExit).grid(row=0, column=3)

#===========================================Calculator Display ===================================================

def btnClick(numbers):
	global operator
	operator = operator +str(numbers)
	text_Input.set(operator)

def btnClear():
	global operator
	operator = ""
	text_Input.set("")

def btnEquals():
	global operator
	sumup = str(eval(operator))
	text_Input.set(sumup)
	operator = ""

txtDisplay = Entry(Cal_F, width=46, bg= "white", bd=4, font=('arial', 12, ),
	justify= RIGHT, textvariable=text_Input)
txtDisplay.grid(row=0, column=0, columnspan=4, pady=1)
txtDisplay.insert(0,"0")



#=========================================== Calculator Buttons===================================================
btn7 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="7", 
	bg="yellow", command=lambda:btnClick(7)).grid(row=2, column=0)

btn8 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="8", 
	bg="yellow",command=lambda:btnClick(8)).grid(row=2, column=1)

btn9 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="9", 
	bg="yellow",command=lambda:btnClick(9)).grid(row=2, column=2)

btnAdd = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="+", 
	bg="yellow",command=lambda:btnClick("+")).grid(row=2, column=3)


#=========================================== Calculator Buttons===================================================
btn4 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="4", 
	bg="yellow",command=lambda:btnClick(4)).grid(row=3, column=0)

btn5 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="5", 
	bg="yellow",command=lambda:btnClick(5)).grid(row=3, column=1)

btn6 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="6", 
	bg="yellow",command=lambda:btnClick(6)).grid(row=3, column=2)

btnSub = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), width=4,  text="-", 
	bg="yellow",command=lambda:btnClick("-")).grid(row=3, column=3)



#=========================================== Calculator Buttons===================================================
btn1 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), 
	width=4,  text="1",bg="yellow", command=lambda:btnClick(1)).grid(row=4, column=0)

btn2 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="2",	bg="yellow",command=lambda:btnClick(2)).grid(row=4, column=1)

btn3 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), 
	width=4,  text="3",	bg="yellow",command=lambda:btnClick(3)).grid(row=4, column=2)

btnMulti = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="x",	bg="yellow",command=lambda:btnClick("*")).grid(row=4, column=3)

#=========================================== Calculator Buttons===================================================
btn0 = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="0",	bg="yellow",command=lambda:btnClick(0)).grid(row=5, column=0)

btnClear = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ), 
	width=4,  text="C",	bg="yellow",command=btnClear).grid(row=5, column=1)

btnEquals = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="=",	bg="yellow",command=btnEquals).grid(row=5, column=2)

btnDiv = Button(Cal_F, padx=16, pady=1, bd=7, fg="black", font=('arial', 16, ),
	width=4,  text="/",	bg="yellow",command=lambda:btnClick("/")).grid(row=5, column=3)


root.mainloop()