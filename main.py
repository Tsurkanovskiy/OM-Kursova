from math import sqrt
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

f_call_counter = 0

def f(x1, x2):
	global f_call_counter
	f_call_counter += 1
	return (10*((x1 - x2)**2) + ((x1 - 1)**2))**4

def create_dir_func(x, direction = [1, 0], func = f):
	def dir_func(h):
		return func(x[0] + direction[0]*h, x[1] + direction[1]*h)
	return dir_func

def module(x1, x2):
	return sqrt((x1**2) + (x2**2))

def reset(x_arr, y_arr):
	global f_call_counter
	f_call_counter = 0
	x_arr, y_arr = [-1.2], [0]
	return x_arr, y_arr

def finish_criteria_grad(dirs, eps, x_arr, y_arr):
	return (module(dirs[0], dirs[1]) >= eps)

def finish_criteria_delta(dirs, eps, x_arr, y_arr):
	if len(x_arr) <= 1:
		return True
	func_delta_crit = ((abs(f(x_arr[-1], y_arr[-1]) - f(x_arr[-2], y_arr[-2]))) > eps)
	x_lenght_delta_crit = ( ((module(x_arr[-1] - x_arr[-2], y_arr[-1] - y_arr[-2])) / (module(x_arr[-2], y_arr[-2]))) > eps)
	return (func_delta_crit|x_lenght_delta_crit)

def Golden_ratio(intr, eps, x, dirs):
	func = create_dir_func(x, dirs)
	a = min(intr)
	b = max(intr)
	x_arr = []
	func_arr = []
	x1 = a + 0.382*(abs(a - b))
	x2 = a + 0.618*(abs(a - b))
	x_arr.extend([a, x1, x2, b])
	func_arr.extend([func(x) for x in x_arr])
	center_index = func_arr.index(min(func_arr))
	while (module(((x[0] + b*dirs[0]) - (x[0] + a*dirs[0])),((x[1] + b*dirs[1])-(x[1] + a*dirs[1]))))>eps:
		x_arr = []
		func_arr = []
		x1 = a + 0.382*(abs(a - b))
		x2 = a + 0.618*(abs(a - b))
		x_arr.extend([a, x1, x2, b])
		func_arr.extend([func(x) for x in x_arr])
		center_index = func_arr.index(min(func_arr))
		try:
			a = x_arr[center_index - 1]
			b = x_arr[center_index + 1]
		except:
			return (a + b)/2
	return x_arr[center_index]

def dsc_Powell(intr, eps, x, dirs):    
    a = min(intr)
    b = max(intr)
    func = create_dir_func(x, dirs)    
    xmin = (a + b) / 2
    f1 = func(a)
    f2 = func(xmin)
    f3 = func(b)
    xApprox = xmin + ((b - xmin) * (f1 - f3)) / (2 * (f1 - 2 * f2 + f3))
    while (abs(xmin - xApprox) >= eps or (abs(func(xmin) - func(xApprox)))>= eps):
        if xApprox < xmin:
            b = xmin
        else:
            a = xmin
        xmin = xApprox
        funcRes = [
            func(a),
            func(xmin),
            func(b),
        ]
        a1 = (funcRes[1] - funcRes[0]) / (xmin - a)
        a2 = ((funcRes[2] - funcRes[0]) / (b - a) - a1) / (b - xmin)
        xApprox = (a + xmin) / 2 - a1 / (2 * a2)

    return xmin

def Sven(x0, delta_lambd, func):
	x_arr = [x0]
	lambd_arr = [func(x0)]
	k = 0
	coef = 1
	if (func(x0)>func(x0 + delta_lambd)):
		coef = 1
		x0 += delta_lambd
		x_arr.append(x0)
		lambd_arr.append(func(x0))
	elif (func(x0)>func((x0 - delta_lambd))):
		coef = -1
		x0 -= delta_lambd
		x_arr.append(x0)
		lambd_arr.append(func(x0))
	else:
		return [-delta_lambd, 0, delta_lambd]
	k += 1
	while (func((coef*delta_lambd*(2**k)))<(func(coef*delta_lambd*(2**(k-1))))):
		x0 += (coef*delta_lambd*(2**k))
		k += 1
		x_arr.append(x0)
		lambd_arr.append(func(x0))
	x_arr.pop(-1)
	lambd_arr.pop(-1)

	x_arr.append((x0 + (coef*delta_lambd*(2**(k-1))))/2)
	lambd_arr.append(func((x0 + (coef*delta_lambd*(2**(k-1))))/2))
	x_arr.append((x0 + (coef*delta_lambd*(2**(k-1)))))
	lambd_arr.append(func((x0 + (coef*delta_lambd*(2**(k-1))))))
	center_index = lambd_arr.index(min(lambd_arr))
	x_result = x_arr[center_index - 1: center_index + 2]
	return x_result









def RRR_diff(func, h):
	Dh = (func(h) - func(- h))/(2*h)
	D2h = (func(2*h) - func(- (2*h)))/(2*h)
	return (4*Dh - D2h)/3

def central_diff(func, h):
	dx = nd.Derivative(func, step=h, method='central')
	return float(dx(0))

def forward_diff(func, h):
	dx = nd.Derivative(func, step=h, method='forward')
	return float(dx(0))

def backward_diff(func, h):
	dx = nd.Derivative(func, step=h, method='backward')
	return float(dx(0))

def get_grad(x, h, diff_func, func = f):
	fun = create_dir_func(x, [1, 0], func)
	dx1 = diff_func(fun, h)
	fun = create_dir_func(x, [0, 1], func)
	dx2 = diff_func(fun, h)
	return [dx1, dx2]


def Gradient_descent(x_arr, y_arr, eps = 0.001, delta_lambd = 0.001, h = 0.02, partan = True, diff_func = RRR_diff, search_method = Golden_ratio, finish_criteria = finish_criteria_delta, restriction = False):
	x = [x_arr[-1], y_arr[-1]]
	dirs = get_grad(x, h, diff_func)
	while finish_criteria(dirs, 0.001, x_arr, y_arr):
		if search_method == dsc_Powell:
			dirs = [i/module(dirs[0], dirs[1]) for i in dirs]
		interval = Sven(0, delta_lambd, create_dir_func(x, dirs))
		lambd = search_method(interval, eps, x, dirs)
		x = [x[0] + lambd*dirs[0], x[1] + lambd*dirs[1]]
		x_arr.append(x[0])
		y_arr.append(x[1])		
		if ((len(x_arr)%3) == 0)&partan:
			dirs = [x[0] - x_arr[-3], x[1] - y_arr[-3]]
		else:
			dirs = get_grad(x, h, diff_func)
	return x_arr, y_arr




x_arr, y_arr = [-1.2], [0]
x_arr, y_arr = Gradient_descent(x_arr, y_arr)

print(f_call_counter)
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()

print("{:15s} {:7d} {:3.4f} {:3.4f}".format("Partan", f_call_counter, x_arr[-1], y_arr[-1]))


x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, partan = False)
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()
print("{:15s} {:7d} {:3.4f} {:3.4f}".format("No partan", f_call_counter, x_arr[-1], y_arr[-1]))



print("{:7s} {:7s} {:6s} {:6s}".format("h step", "counter", "x1", "x2"))
values_arr = [0.02, 0.01, 0.001, 0.0001, 0.00001]
counter_arr = []
for val in values_arr:
	x_arr, y_arr = reset(x_arr, y_arr)
	x_arr, y_arr = Gradient_descent(x_arr, y_arr, h = val)
	print("{:7s} {:7d} {:3.4f} {:3.4f}".format(str(val), f_call_counter, x_arr[-1], y_arr[-1]))
	counter_arr.append(f_call_counter)



values_arr = [RRR_diff, central_diff, forward_diff, backward_diff]
counter_arr = []
print()
print("{:15s} {:7s} {:6s} {:6s}".format("diff method", "counter", "x1", "x2"))
for val in values_arr:
	x_arr, y_arr = reset(x_arr, y_arr)
	x_arr, y_arr = Gradient_descent(x_arr, y_arr, h = 0.001, diff_func = val)
	print("{:15s} {:7d} {:3.4f} {:3.4f}".format(val.__name__, f_call_counter, x_arr[-1], y_arr[-1]))






x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, h = 0.001, diff_func = central_diff)
print("\n{:15s} {:7d} {:3.4f} {:3.4f}".format("Golden Ratio", f_call_counter, x_arr[-1], y_arr[-1]))
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()

x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, h = 0.001, diff_func = central_diff, search_method = dsc_Powell)
print("{:15s} {:7d} {:3.4f} {:3.4f}".format("DSC-Powell", f_call_counter, x_arr[-1], y_arr[-1]))
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()



print("\n{:7s} {:7s} {:6s} {:6s}".format("epsilon", "counter", "x1", "x2"))
values_arr = [0.5, 0.1, 0.01, 0.001, 0.0001]
counter_arr = []
for val in values_arr:
	x_arr, y_arr = reset(x_arr, y_arr)
	x_arr, y_arr = Gradient_descent(x_arr, y_arr, eps = val, h = 0.001, diff_func = central_diff, search_method = dsc_Powell)
	print("{:7s} {:7d} {:3.4f} {:3.4f}".format(str(val), f_call_counter, x_arr[-1], y_arr[-1]))
	counter_arr.append(f_call_counter)


print("\n{:10s} {:7s} {:6s} {:6s}".format("Sven step", "counter", "x1", "x2"))
values_arr = [0.5, 0.1, 0.01, 0.001, 0.0001]
counter_arr = []
for val in values_arr:
	x_arr, y_arr = reset(x_arr, y_arr)
	x_arr, y_arr = Gradient_descent(x_arr, y_arr, eps = 0.0001, delta_lambd = val, h = 0.001, diff_func = central_diff, search_method = dsc_Powell)
	print("{:10s} {:7d} {:3.4f} {:3.4f}".format(str(val), f_call_counter, x_arr[-1], y_arr[-1]))
	counter_arr.append(f_call_counter)



x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, eps = 0.0001, delta_lambd = 0.5, h = 0.001, diff_func = central_diff, search_method = dsc_Powell)
print("\n{:15s} {:7d} {:3.4f} {:3.4f}".format("Module", f_call_counter, x_arr[-1], y_arr[-1]))
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()

x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, eps = 0.0001, delta_lambd = 0.5, h = 0.001, diff_func = central_diff, search_method = dsc_Powell, finish_criteria = finish_criteria_grad)
print("{:15s} {:7d} {:3.4f} {:3.4f}".format("Gradient", f_call_counter, x_arr[-1], y_arr[-1]))
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()


x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent(x_arr, y_arr, eps = 0.0001, delta_lambd = 0.5, h = 0.001, partan = False, diff_func = central_diff, search_method = dsc_Powell, finish_criteria = finish_criteria_grad)
print("{:15s} {:7d} {:3.4f} {:3.4f}".format("No partan", f_call_counter, x_arr[-1], y_arr[-1]))
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
dot = plt.scatter(1, 1)
plt.legend([dot, search], [ "Точка мінімуму", "Прямі пошуку"])
plt.show()





def restriction_linear(x1):
	return 0.5 - 0.8*x1 

r1_arr = np.linspace(-1.3, 1.25, 20)
r2_arr = [restriction_linear(r1) for r1 in r1_arr]


R = 0


def restriction(x1, x2):
	return 0.8*x1 + x2 - 0.5

def outer_point(x1, x2, R = 1):
	return f(x1, x2) + R*(0.8*x1 + x2 - 0.5)

def set_R(r):
	def R_set(x1, x2):
		return outer_point(x1, x2, r)
	return R_set

def create_dir_func(x, direction = [1, 0], func = set_R(R)):
	func = set_R(R)
	def dir_func(h):
		return func(x[0] + direction[0]*h, x[1] + direction[1]*h)
	return dir_func

def Gradient_descent_restricted(x_arr, y_arr, partan = True, diff_func = RRR_diff, R_var = 100):
	eps = 0.0001
	delta_lambd = 0.5
	h = 0.001
	search_method = dsc_Powell
	finish_criteria = finish_criteria_grad
	diff_func = central_diff
	global R


	x = [x_arr[-1], y_arr[-1]]
	if (restriction(x[0],x[1])>0):
		R = R_var
	else:
		R = 0
	dirs = get_grad(x, h, diff_func)
	while finish_criteria(dirs, 1, x_arr, y_arr)|(restriction(x[0],x[1])>0):
		if search_method == dsc_Powell:
			dirs = [i/module(dirs[0], dirs[1]) for i in dirs]
		interval = Sven(0, delta_lambd, create_dir_func(x, dirs))
		lambd = search_method(interval, eps, x, dirs)
		x = [x[0] + lambd*dirs[0], x[1] + lambd*dirs[1]]
		x_arr.append(x[0])
		y_arr.append(x[1])		
		if ((len(x_arr)%3) == 0)&partan:
			dirs = [x[0] - x_arr[-3], x[1] - y_arr[-3]]
		else:
			dirs = get_grad(x, h, diff_func)
		if (restriction(x[0],x[1])>0):
			R = R_var
		else:
			R = 0
	return x_arr, y_arr


R_list = [1, 10, 100, 1000]
print("\n{:5s} {:7s} {:6s} {:6s}".format("R", "counter", "x1", "x2"))
for R_var in R_list:
	x_arr, y_arr = reset(x_arr, y_arr)
	x_arr, y_arr = Gradient_descent_restricted(x_arr, y_arr, R_var = R_var)
	print("{:5s} {:7d} {:3.4f} {:3.4f}".format(str(R_var), f_call_counter, x_arr[-1], y_arr[-1]))

x_arr, y_arr = reset(x_arr, y_arr)
x_arr, y_arr = Gradient_descent_restricted(x_arr, y_arr, R_var = 10)
plt.grid(True)
search, = plt.plot(x_arr, y_arr, color="orange")
restriction_plot, = plt.plot(r1_arr, r2_arr, color="red")
dot = plt.scatter(1, 1)
plt.legend([dot, restriction_plot, search], ["Точка мінімуму", "Обмеження", "Прямі пошуку"])
plt.show()